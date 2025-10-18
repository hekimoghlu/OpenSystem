all: papers.dmg


papers.dmg: papers.app
	sh dmg.sh

papers.app: papers.bundle Info.plist Papers.icns launcher.sh
	rm -rf papers.app
	gtk-mac-bundler papers.bundle

clean:
	rm -rf papers.app papers.dmg


.PHONY: all clean
