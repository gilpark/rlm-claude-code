#!/usr/bin/env npx ts-node
/**
 * download-wheel.ts
 *
 * Download pre-built Python wheel from GitHub releases.
 * Only downloads if the release version matches package.json version.
 */

import fs from 'fs';
import path from 'path';
import https from 'https';
import http from 'http';
import { ROOT_DIR, log, getPlatform } from './common';

interface GitHubRelease {
  tag_name: string;
  assets: Array<{
    name: string;
    browser_download_url: string;
  }>;
}

function getPackageVersion(): string {
  const packageJsonPath = path.join(ROOT_DIR, 'package.json');
  const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
  return packageJson.version;
}

function downloadFile(url: string, destPath: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const protocol = url.startsWith('https') ? https : http;
    const file = fs.createWriteStream(destPath);

    protocol.get(url, (response) => {
      if (response.statusCode === 301 || response.statusCode === 302) {
        const redirectUrl = response.headers.location;
        if (redirectUrl) {
          downloadFile(redirectUrl, destPath).then(resolve).catch(reject);
          return;
        }
      }

      if (response.statusCode !== 200) {
        reject(new Error(`HTTP ${response.statusCode}: ${url}`));
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
  });
}

async function getLatestRelease(): Promise<GitHubRelease> {
  return new Promise((resolve, reject) => {
    const options = {
      hostname: 'api.github.com',
      path: '/repos/rand/rlm-claude-code/releases/latest',
      headers: {
        'User-Agent': 'rlm-claude-code-installer',
      },
    };

    https.get(options, (response) => {
      let data = '';
      response.on('data', (chunk) => (data += chunk));
      response.on('end', () => {
        try {
          const release = JSON.parse(data) as GitHubRelease;
          resolve(release);
        } catch (e) {
          reject(e);
        }
      });
    }).on('error', reject);
  });
}

function getWheelPlatform(os: string, cpu: string): string {
  if (os === 'darwin' && cpu === 'arm64') {
    return 'macosx_11_0_arm64';
  } else if (os === 'darwin' && cpu === 'amd64') {
    return 'macosx_10_9_x86_64';
  } else if (os === 'linux' && cpu === 'amd64') {
    return 'manylinux_2_17_x86_64';
  } else if (os === 'linux' && cpu === 'arm64') {
    return 'manylinux_2_17_aarch64';
  } else if (os === 'windows') {
    return 'win_amd64';
  }
  return 'unknown';
}

async function main(): Promise<void> {
  log('\n=== Downloading Python Wheel ===', 'cyan');

  const { os, cpu } = getPlatform();
  const wheelPlatform = getWheelPlatform(os, cpu);
  const packageVersion = getPackageVersion();

  log(`Platform: ${wheelPlatform}`, 'dim');
  log(`Package version: ${packageVersion}`, 'dim');

  try {
    log('Fetching latest release information...');
    const release = await getLatestRelease();
    const releaseVersion = release.tag_name.replace(/^v/, ''); // Remove 'v' prefix

    // Check version compatibility
    if (releaseVersion !== packageVersion) {
      log(`\nVersion mismatch: package=${packageVersion}, release=${releaseVersion}`, 'yellow');
      log('Cannot download wheel - versions must match for compatibility.', 'yellow');
      log('Falling back to building from source...', 'yellow');
      process.exit(1);
    }

    // Find matching wheel (rlm_core, not rlm_claude_code)
    const wheelAsset = release.assets.find((a) => {
      return (
        a.name.includes('rlm_core') &&
        a.name.endsWith('.whl') &&
        (a.name.includes(wheelPlatform) || a.name.includes('py3-none-any'))
      );
    });

    if (!wheelAsset) {
      log(`\nNo pre-built wheel found for ${wheelPlatform}`, 'yellow');
      log('Will need to build from source.', 'yellow');
      process.exit(1);
    }

    const wheelPath = path.join(ROOT_DIR, wheelAsset.name);
    log(`Downloading ${wheelAsset.name}...`);

    await downloadFile(wheelAsset.browser_download_url, wheelPath);

    log(`Wheel downloaded: ${wheelAsset.name}`, 'green');
    log('Install with: uv pip install ' + wheelAsset.name, 'dim');

  } catch (error) {
    const err = error as Error;
    log(`Download failed: ${err.message}`, 'red');
    process.exit(1);
  }
}

main();
