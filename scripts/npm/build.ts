#!/usr/bin/env npx ts-node
/**
 * build.ts
 *
 * Build script for RLM-Claude-Code.
 * Supports both downloading pre-built binaries and building from source.
 *
 * Usage:
 *   npm run build              # Download pre-built binaries (default)
 *   npm run build -- --all     # Build everything from source
 *   npm run build -- --binaries-only  # Build only Go binaries
 *   npm run build -- --wheel-only     # Build only rlm-core wheel
 */

import fs from 'fs';
import path from 'path';
import { ROOT_DIR, log, runCommand, fileExists, getPlatform } from './common';

interface BuildOptions {
  all: boolean;
  develop: boolean;
  binariesOnly: boolean;
  wheelOnly: boolean;
}

function parseArgs(): BuildOptions {
  return {
    all: process.argv.includes('--all'),
    develop: process.argv.includes('--develop'),
    binariesOnly: process.argv.includes('--binaries-only'),
    wheelOnly: process.argv.includes('--wheel-only'),
  };
}

function buildWheelFromSource(): boolean {
  log('\n=== Building rlm_core Wheel from Source ===', 'cyan');

  if (!fileExists('vendor/loop/rlm-core/Cargo.toml')) {
    log('rlm-core submodule not found. Run: git submodule update --init --recursive', 'red');
    return false;
  }

  try {
    // Build rlm_core wheel from submodule (creates rlm_core-0.1.0-...whl)
    log('Building rlm_core wheel with maturin build...');
    runCommand('uv run maturin build --release', { cwd: path.join(ROOT_DIR, 'vendor/loop/rlm-core') });

    // Find and install the wheel directly from build location
    const wheelDir = path.join(ROOT_DIR, 'vendor/loop/rlm-core/target/wheels');
    if (fs.existsSync(wheelDir)) {
      const wheels = fs.readdirSync(wheelDir).filter(f => f.startsWith('rlm_core') && f.endsWith('.whl'));
      if (wheels.length > 0) {
        const wheelPath = path.join(wheelDir, wheels[0]);
        log(`Wheel built: ${wheels[0]}`, 'green');

        // Install the wheel directly
        log('Installing rlm_core wheel...');
        runCommand(`uv run pip install --force-reinstall ${wheelPath}`);
        log('rlm_core wheel installed.', 'green');
      }
    }

    return true;
  } catch (error) {
    const err = error as Error;
    log(`Error building wheel: ${err.message}`, 'red');
    return false;
  }
}

function developWheel(): boolean {
  log('\n=== Developing rlm_core (maturin develop) ===', 'cyan');

  if (!fileExists('vendor/loop/rlm-core/Cargo.toml')) {
    log('rlm-core submodule not found. Run: git submodule update --init --recursive', 'red');
    return false;
  }

  try {
    // maturin develop builds and installs directly - no wheel file created
    // Fast iteration for development
    log('Building and installing rlm_core with maturin develop...');
    runCommand('uv run maturin develop --release', { cwd: path.join(ROOT_DIR, 'vendor/loop/rlm-core') });
    log('rlm_core installed in development mode.', 'green');
    log('You can now: import rlm_core', 'dim');
    return true;
  } catch (error) {
    const err = error as Error;
    log(`Error developing wheel: ${err.message}`, 'red');
    return false;
  }
}

function buildGoBinaries(): boolean {
  log('\n=== Building Go Hook Binaries ===', 'cyan');

  if (!fileExists('Makefile')) {
    log('Makefile not found.', 'red');
    return false;
  }

  try {
    log('Building Go binaries for all platforms with Make...');
    runCommand(`make all`);
    log('Go binaries built successfully.', 'green');
    return true;
  } catch (error) {
    const err = error as Error;
    log(`Error building binaries: ${err.message}`, 'red');
    return false;
  }
}

async function downloadBinaries(): Promise<boolean> {
  log('\n=== Downloading Pre-built Binaries ===', 'cyan');

  const { os, platform } = getPlatform();
  const binDir = path.join(ROOT_DIR, 'bin');

  if (!fs.existsSync(binDir)) {
    fs.mkdirSync(binDir, { recursive: true });
  }

  const binaries = ['session-init', 'complexity-check', 'trajectory-save'];
  const expectedBinary = os === 'windows'
    ? `${binaries[0]}-${platform}.exe`
    : `${binaries[0]}-${platform}`;

  if (fs.existsSync(path.join(binDir, expectedBinary))) {
    log(`Binaries already exist for ${platform}.`, 'green');
    return true;
  }

  log('Downloading binaries...');
  runCommand('npx ts-node scripts/npm/download-binaries.ts', { silent: false });
  return true;
}

async function downloadWheel(): Promise<boolean> {
  log('\n=== Downloading Pre-built Wheel ===', 'cyan');
  runCommand('npx ts-node scripts/npm/download-wheel.ts', { silent: false });
  return true;
}

function installDeps(): boolean {
  log('\n=== Installing Dependencies ===', 'cyan');

  try {
    runCommand('uv sync');
    log('Dependencies installed.', 'green');
    return true;
  } catch (error) {
    const err = error as Error;
    log(`Error installing dependencies: ${err.message}`, 'red');
    return false;
  }
}

async function main(): Promise<void> {
  log('\n=========================================', 'cyan');
  log('  RLM-Claude-Code Build', 'cyan');
  log('=========================================', 'cyan');

  const options = parseArgs();

  if (options.all) {
    log('Building everything from source...', 'cyan');

    if (!buildWheelFromSource()) {
      process.exit(1);
    }

    if (!buildGoBinaries()) {
      process.exit(1);
    }

    if (!installDeps()) {
      process.exit(1);
    }
  } else if (options.develop) {
    log('Development mode: fast iteration with maturin develop...', 'cyan');

    if (!developWheel()) {
      process.exit(1);
    }

    if (!buildGoBinaries()) {
      process.exit(1);
    }

    if (!installDeps()) {
      process.exit(1);
    }
  } else if (options.binariesOnly) {
    if (!buildGoBinaries()) {
      process.exit(1);
    }
  } else if (options.wheelOnly) {
    if (!buildWheelFromSource()) {
      process.exit(1);
    }
  } else {
    // Default: download pre-built
    await downloadBinaries();
    await downloadWheel();
    installDeps();
  }

  log('\n=========================================', 'green');
  log('  Build Complete!', 'green');
  log('=========================================', 'green');
  log('\nNext: npm run verify');
}

main().catch((error) => {
  const err = error as Error;
  log(`\nBuild failed: ${err.message}`, 'red');
  process.exit(1);
});
