AUTOMAKE_OPTIONS = foreign

SUBDIRS = src

if WITH_GNOME
SUBDIRS += properties po

if WITH_AUTHDLG
SUBDIRS += auth-dialog
endif
endif

dbusservicedir = $(datadir)/dbus-1/system.d
dbusservice_DATA = nm-iodine-service.conf

nmvpnservicedir = $(NM_VPN_SERVICE_DIR)
nmvpnservice_DATA = nm-iodine-service.name

appdatadir = $(datadir)/appdata
appdata_DATA = $(appdata_in_files:.xml.in=.xml)
appdata_in_files = appdata/network-manager-iodine.metainfo.xml.in
@INTLTOOL_XML_RULE@

nm-iodine-service.name: $(srcdir)/nm-iodine-service.name.in
	sed -e 's|[@]LIBEXECDIR[@]|$(libexecdir)|g' \
	    -e 's|[@]PLUGINDIR[@]/|@NM_PLUGIN_DIR@|g' \
	    $< >$@

DISTCHECK_CONFIGURE_FLAGS = --enable-more-warnings=yes

EXTRA_DIST = nm-iodine-service.name.in \
             $(dbusservice_DATA)  \
             $(desktop_in_files)  \
             $(icon_DATA)         \
             $(appdata_in_files)  \
             $(appdata_DATA)      \
             intltool-extract.in  \
             intltool-merge.in    \
             intltool-update.in

CLEANFILES = $(nmvpnservice_DATA) \
             $(desktop_DATA)      \
             $(appdata_DATA)

DISTCLEANFILES = intltool-extract intltool-merge intltool-update

ACLOCAL_AMFLAGS = -I m4
