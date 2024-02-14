# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['Snap2LaTeX.py'],
    pathex=[],
    binaries=[],
    datas=[("icon.png", ".")],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Snap2LaTeX',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Snap2LaTeX',
)
app = BUNDLE(
    coll,
    name='Snap2LaTeX.app',
    icon='icon.icns',
    bundle_identifier="me.fanjiang.Snap2LaTeX",
    info_plist={
        'LSUIElement': True,
        'LSBackgroundOnly': True,
        'NSUIElement': True
    },
)
