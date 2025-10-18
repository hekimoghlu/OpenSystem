#!/bin/sh

test -f papers.dmg && rm papers.dmg
create-dmg \
  --volname "Papers Installer" \
  --volicon "Papers.icns" \
  --window-pos 200 120 \
  --window-size 800 400 \
  --icon-size 100 \
  --icon "papers.app" 200 190 \
  --hide-extension "papers.app" \
  --app-drop-link 600 185 \
  "papers.dmg" \
  "papers.app/"
