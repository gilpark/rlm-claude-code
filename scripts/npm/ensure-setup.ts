#!/usr/bin/env npx ts-node
/**
 * ensure-setup.ts
 *
 * Smart self-healing setup script for rlm-claude-code plugin.
 * Single entry point that handles everything:
 *
 * 1. Read version from marketplace.json
 * 2. Check GitHub releases for matching version
 * 3. If release exists: download binaries + wheel
 * 4. If no release: build from source
 * 5. Install Python dependencies
 * 6. Output JSON status for AI agent
 *
 * Usage:
 *   ensure-setup.ts           # Check and auto-fix
 *   ensure-setup.ts --json    # Output JSON for hooks
 *   ensure-setup.ts --check   # Only check, don't fix
 */

import fs from 'fs';
import path from 'path';
import https from 'https';
import { execSync } from 'child_process';
import * as tar from 'tar';
import { ROOT_DIR, log, getPlatform, runCommand } from './common';

// Types
type InstallMode = 'marketplace' | 'dev';
type Status = 'ok' | 'missing' | 'partial' | 'no-venv' | 'error';

interface SetupStatus {
  platform: string;
  version: string;
  mode: InstallMode;
  release: 'found' | 'not-found' | 'error';
  uv: Status;
  venv: Status;
  binaries: Status;
  rlmCore: Status;
  needsAttention: boolean;
  action: string;
  instructions: string[];
}

interface GitHubRelease {
  tag_name: string;
  assets: Array<{
    name: string;
    browser_download_url: string;
  }>;
}

// ============================================================================
// Version Detection
// ============================================================================

function getVersion(): string {
  // Try marketplace.json first
  const marketplacePath = path.join(ROOT_DIR, '.claude-plugin', 'marketplace.json');
  if (fs.existsSync(marketplacePath)) {
    const marketplace = JSON.parse(fs.readFileSync(marketplacePath, 'utf8'));
    if (marketplace.plugins?.[0]?.version) {
      return marketplace.plugins[0].version;
    }
  }

  // Fallback to package.json
  const packagePath = path.join(ROOT_DIR, 'package.json');
  if (fs.existsSync(packagePath)) {
    const pkg = JSON.parse(fs.readFileSync(packagePath, 'utf8'));
    return pkg.version;
  }

  return 'unknown';
}

function detectInstallMode(): InstallMode {
  if (ROOT_DIR.includes('/dev/') || ROOT_DIR.endsWith('/dev')) {
    return 'dev';
  }
  if (fs.existsSync(path.join(ROOT_DIR, '.git'))) {
    return 'dev';
  }
  return 'marketplace';
}

// ============================================================================
// GitHub Release Check
// ============================================================================

async function getMatchingRelease(version: string): Promise<GitHubRelease | null> {
  return new Promise((resolve) => {
    const options = {
      hostname: 'api.github.com',
      path: `/repos/rand/rlm-claude-code/releases/tags/v${version}`,
      headers: {
        'User-Agent': 'rlm-claude-code-installer',
      },
      timeout: 10000,
    };

    https.get(options, (response) => {
      let data = '';
      response.on('data', (chunk) => (data += chunk));
      response.on('end', () => {
        if (response.statusCode === 200) {
          try {
            resolve(JSON.parse(data) as GitHubRelease);
          } catch {
            resolve(null);
          }
        } else {
          resolve(null);
        }
      });
    }).on('error', () => resolve(null))
      .on('timeout', () => resolve(null));
  });
}

// ============================================================================
// Status Checks
// ============================================================================

function checkUv(): Status {
  try {
    execSync('uv --version', { stdio: 'pipe' });
    return 'ok';
  } catch {
    return 'missing';
  }
}

function checkVenv(): Status {
  const venvPath = process.platform === 'win32'
    ? path.join(ROOT_DIR, '.venv', 'Scripts', 'activate')
    : path.join(ROOT_DIR, '.venv', 'bin', 'activate');
  return fs.existsSync(venvPath) ? 'ok' : 'missing';
}

function checkBinaries(): Status {
  const { os, cpu } = getPlatform();
  const binDir = path.join(ROOT_DIR, 'bin');

  if (!fs.existsSync(binDir)) {
    return 'missing';
  }

  const binaries = ['session-init', 'complexity-check', 'trajectory-save'];
  const expected = binaries.map(b =>
    os === 'windows' ? `${b}-${os}-${cpu}.exe` : `${b}-${os}-${cpu}`
  );

  let found = 0;
  for (const binary of expected) {
    if (fs.existsSync(path.join(binDir, binary))) {
      found++;
    }
  }

  if (found === 0) return 'missing';
  if (found < expected.length) return 'partial';
  return 'ok';
}

function checkRlmCore(): Status {
  if (checkVenv() !== 'ok') {
    return 'no-venv';
  }

  const venvPython = process.platform === 'win32'
    ? path.join(ROOT_DIR, '.venv', 'Scripts', 'python.exe')
    : path.join(ROOT_DIR, '.venv', 'bin', 'python');

  if (!fs.existsSync(venvPython)) {
    return 'no-venv';
  }

  try {
    execSync(`"${venvPython}" -c "import rlm_core"`, { stdio: 'pipe' });
    return 'ok';
  } catch {
    return 'missing';
  }
}

// ============================================================================
// Fix Functions (Download)
// ============================================================================

async function downloadFile(url: string, destPath: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(destPath);
    const doDownload = (downloadUrl: string) => {
      https.get(downloadUrl, (response) => {
        if (response.statusCode === 301 || response.statusCode === 302) {
          const redirectUrl = response.headers.location;
          if (redirectUrl) {
            doDownload(redirectUrl);
            return;
          }
        }
        if (response.statusCode !== 200) {
          reject(new Error(`HTTP ${response.statusCode}`));
          return;
        }
        response.pipe(file);
        file.on('finish', () => {
          file.close();
          resolve();
        });
      }).on('error', (err) => {
        fs.unlink(destPath, () => {});
        reject(err);
      });
    };
    doDownload(url);
  });
}

async function downloadBinariesFromRelease(release: GitHubRelease, version: string): Promise<boolean> {
  const { os, cpu } = getPlatform();
  const binDir = path.join(ROOT_DIR, 'bin');
  const archiveName = `hooks-v${version}-${os}-${cpu}.tar.gz`;

  const asset = release.assets.find(a => a.name === archiveName);
  if (!asset) {
    log(`  No binaries archive found: ${archiveName}`, 'yellow');
    return false;
  }

  if (!fs.existsSync(binDir)) {
    fs.mkdirSync(binDir, { recursive: true });
  }

  const archivePath = path.join(binDir, archiveName);
  log(`  Downloading ${archiveName}...`);

  try {
    await downloadFile(asset.browser_download_url, archivePath);
    await tar.x({ file: archivePath, cwd: binDir });
    fs.unlinkSync(archivePath);

    // Set permissions on Unix
    if (process.platform !== 'win32') {
      const binaries = fs.readdirSync(binDir).filter(f =>
        f.startsWith('session-init') || f.startsWith('complexity-check') || f.startsWith('trajectory-save')
      );
      for (const binary of binaries) {
        fs.chmodSync(path.join(binDir, binary), 0o755);
      }
    }

    // Remove Apple quarantine on macOS
    if (process.platform === 'darwin') {
      try {
        execSync(`xattr -cr "${binDir}"`, { stdio: 'pipe' });
      } catch { /* ignore */ }
    }

    log(`  Binaries downloaded`, 'green');
    return true;
  } catch (error) {
    const err = error as Error;
    log(`  Download failed: ${err.message}`, 'red');
    return false;
  }
}

async function downloadWheelFromRelease(release: GitHubRelease): Promise<boolean> {
  const { os, cpu } = getPlatform();

  // Map platform to wheel platform
  let wheelPlatform: string;
  if (os === 'darwin' && cpu === 'arm64') {
    wheelPlatform = 'macosx_11_0_arm64';
  } else if (os === 'darwin' && cpu === 'amd64') {
    wheelPlatform = 'macosx_10_9_x86_64';
  } else if (os === 'linux' && cpu === 'amd64') {
    wheelPlatform = 'manylinux_2_17_x86_64';
  } else if (os === 'linux' && cpu === 'arm64') {
    wheelPlatform = 'manylinux_2_17_aarch64';
  } else if (os === 'windows') {
    wheelPlatform = 'win_amd64';
  } else {
    wheelPlatform = 'unknown';
  }

  // Find matching wheel (rlm_core)
  const wheelAsset = release.assets.find(a =>
    a.name.includes('rlm_core') &&
    a.name.endsWith('.whl') &&
    (a.name.includes(wheelPlatform) || a.name.includes('py3-none-any'))
  );

  if (!wheelAsset) {
    log(`  No wheel found for ${wheelPlatform}`, 'yellow');
    return false;
  }

  const wheelPath = path.join(ROOT_DIR, wheelAsset.name);
  log(`  Downloading ${wheelAsset.name}...`);

  try {
    await downloadFile(wheelAsset.browser_download_url, wheelPath);
    log(`  Installing wheel...`);
    runCommand(`uv pip install --force-reinstall "${wheelPath}"`, { silent: true });
    fs.unlinkSync(wheelPath);
    log(`  Wheel installed`, 'green');
    return true;
  } catch (error) {
    const err = error as Error;
    log(`  Wheel download/install failed: ${err.message}`, 'red');
    return false;
  }
}

// ============================================================================
// Fix Functions (Build from Source)
// ============================================================================

function buildBinariesFromSource(): boolean {
  log('  Building Go binaries from source...');

  if (!fs.existsSync(path.join(ROOT_DIR, 'Makefile'))) {
    log('  Makefile not found', 'red');
    return false;
  }

  try {
    runCommand('make all', { silent: true });
    log('  Binaries built', 'green');
    return true;
  } catch (error) {
    const err = error as Error;
    log(`  Build failed: ${err.message}`, 'red');
    return false;
  }
}

function buildWheelFromSource(): boolean {
  log('  Building rlm_core wheel from source...');

  const rlmCorePath = path.join(ROOT_DIR, 'vendor/loop/rlm-core/Cargo.toml');
  if (!fs.existsSync(rlmCorePath)) {
    log('  rlm-core submodule not found', 'red');
    log('  Run: git submodule update --init --recursive', 'yellow');
    return false;
  }

  try {
    runCommand('uv run maturin build --release', {
      cwd: path.join(ROOT_DIR, 'vendor/loop/rlm-core'),
      silent: true
    });

    // Find and install the wheel
    const wheelDir = path.join(ROOT_DIR, 'vendor/loop/rlm-core/target/wheels');
    if (fs.existsSync(wheelDir)) {
      const wheels = fs.readdirSync(wheelDir).filter(f => f.startsWith('rlm_core') && f.endsWith('.whl'));
      if (wheels.length > 0) {
        const wheelPath = path.join(wheelDir, wheels[0]);
        runCommand(`uv pip install --force-reinstall "${wheelPath}"`, { silent: true });
        log(`  Wheel built and installed`, 'green');
        return true;
      }
    }

    log('  Wheel not found after build', 'red');
    return false;
  } catch (error) {
    const err = error as Error;
    log(`  Build failed: ${err.message}`, 'red');
    return false;
  }
}

function createVenv(): boolean {
  log('  Creating virtual environment...');

  try {
    runCommand('uv venv', { silent: true });
    runCommand('uv sync', { silent: true });
    log('  venv created', 'green');
    return true;
  } catch (error) {
    const err = error as Error;
    log(`  Failed: ${err.message}`, 'red');
    return false;
  }
}

function syncDeps(): boolean {
  log('  Syncing Python dependencies...');

  try {
    runCommand('uv sync', { silent: true });
    log('  Dependencies synced', 'green');
    return true;
  } catch (error) {
    const err = error as Error;
    log(`  Sync failed: ${err.message}`, 'red');
    return false;
  }
}

// ============================================================================
// Main Logic
// ============================================================================

async function ensureSetup(checkOnly: boolean): Promise<SetupStatus> {
  const { platform } = getPlatform();
  const version = getVersion();
  const mode = detectInstallMode();

  // Check current status
  let uv = checkUv();
  let venv = checkVenv();
  let binaries = checkBinaries();
  let rlmCore = checkRlmCore();

  // Check for matching release
  let release: 'found' | 'not-found' | 'error' = 'not-found';
  let githubRelease: GitHubRelease | null = null;

  if (uv === 'ok') {
    githubRelease = await getMatchingRelease(version);
    release = githubRelease ? 'found' : 'not-found';
  }

  // Determine action
  const needsFix = venv !== 'ok' || binaries !== 'ok' || rlmCore !== 'ok';

  if (!checkOnly && needsFix) {
    log('\n=== Fixing Setup Issues ===', 'cyan');

    // 1. Create venv if needed
    if (venv !== 'ok') {
      if (uv !== 'ok') {
        log('  uv is not installed!', 'red');
        log('  Install: curl -LsSf https://astral.sh/uv/install.sh | sh', 'yellow');
      } else {
        createVenv();
        venv = checkVenv();
      }
    }

    // 2. Get binaries
    if (binaries !== 'ok' && venv === 'ok') {
      if (release === 'found' && githubRelease) {
        await downloadBinariesFromRelease(githubRelease, version);
      } else {
        buildBinariesFromSource();
      }
      binaries = checkBinaries();
    }

    // 3. Get rlm_core wheel
    if (rlmCore !== 'ok' && venv === 'ok') {
      if (release === 'found' && githubRelease) {
        const downloaded = await downloadWheelFromRelease(githubRelease);
        if (!downloaded) {
          // Fallback to source build
          buildWheelFromSource();
        }
      } else {
        buildWheelFromSource();
      }
      rlmCore = checkRlmCore();
    }

    // 4. Sync dependencies
    if (venv === 'ok') {
      syncDeps();
    }
  }

  // Final status
  const needsAttention = venv !== 'ok' || binaries !== 'ok' || rlmCore !== 'ok';

  const instructions: string[] = [];
  let action = '';

  if (uv !== 'ok') {
    instructions.push('Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh');
    action = 'install-uv';
  } else if (venv !== 'ok') {
    instructions.push('Run: npm run ensure-setup');
    action = 'create-venv';
  } else if (binaries !== 'ok' || rlmCore !== 'ok') {
    if (release === 'found') {
      instructions.push('Run: npm run ensure-setup');
      action = 'download';
    } else {
      instructions.push('Run: npm run build (build from source)');
      action = 'build';
    }
  } else {
    action = 'ready';
  }

  return {
    platform,
    version,
    mode,
    release,
    uv,
    venv,
    binaries,
    rlmCore,
    needsAttention,
    action,
    instructions,
  };
}

function outputJson(status: SetupStatus): void {
  let additionalContext: string;

  if (status.needsAttention) {
    const issues: string[] = [];
    if (status.uv !== 'ok') issues.push('uv package manager missing');
    if (status.venv !== 'ok') issues.push('Python venv missing');
    if (status.binaries !== 'ok') issues.push('Hook binaries missing');
    if (status.rlmCore !== 'ok') issues.push('rlm_core package missing');

    additionalContext = `RLM setup needs attention: ${issues.join(', ')}. `;
    additionalContext += `Version ${status.version}, release ${status.release}. `;

    if (status.action === 'install-uv') {
      additionalContext += 'Install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh';
    } else if (status.instructions.length > 0) {
      additionalContext += `Fix: ${status.instructions[0]}`;
    }
  } else {
    additionalContext = `RLM plugin ready (v${status.version}). All checks passed.`;
  }

  const output = {
    hookSpecificOutput: {
      hookEventName: 'EnsureSetup',
      additionalContext,
      rlmSetupStatus: status,
    }
  };

  console.log(JSON.stringify(output));
}

function outputText(status: SetupStatus): void {
  log('\n=========================================', 'cyan');
  log('  RLM Plugin Setup Status', 'cyan');
  log('=========================================', 'cyan');

  log(`\nVersion:  ${status.version}`, 'dim');
  log(`Platform: ${status.platform}`, 'dim');
  log(`Mode:     ${status.mode}`, 'dim');
  log(`Release:  ${status.release}`, status.release === 'found' ? 'green' : 'yellow');

  log('\nChecks:', 'yellow');
  log(`  uv:       ${status.uv}`, status.uv === 'ok' ? 'green' : 'red');
  log(`  venv:     ${status.venv}`, status.venv === 'ok' ? 'green' : 'red');
  log(`  binaries: ${status.binaries}`, status.binaries === 'ok' ? 'green' : 'yellow');
  log(`  rlm_core: ${status.rlmCore}`, status.rlmCore === 'ok' ? 'green' : 'red');

  if (status.needsAttention) {
    log('\nAction Required:', 'yellow');
    for (const instruction of status.instructions) {
      log(`  ${instruction}`, 'cyan');
    }
  } else {
    log('\nAll checks passed!', 'green');
  }
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const jsonOutput = args.includes('--json');
  const checkOnly = args.includes('--check');

  const status = await ensureSetup(checkOnly);

  if (jsonOutput) {
    outputJson(status);
  } else {
    outputText(status);
  }

  process.exit(status.needsAttention ? 1 : 0);
}

main().catch((error) => {
  console.error(`Setup failed: ${(error as Error).message}`);
  process.exit(1);
});
