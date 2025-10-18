AUTOMAKE_OPTIONS = subdir-objects

DISTCHECK_CONFIGURE_FLAGS = \
	--enable-more-warnings=yes \
	--with-gtk4

libexec_PROGRAMS =

noinst_LTLIBRARIES =

EXTRA_DIST =

CLEANFILES =

noinst_PROGRAMS =

SUBDIRS = \
	. \
	po

plugindir = $(libdir)/NetworkManager
plugin_LTLIBRARIES =

dbusservicedir = $(datadir)/dbus-1/system.d
dbusservice_DATA = nm-pptp-service.conf

nmvpnservicedir = $(NM_VPN_SERVICE_DIR)
nmvpnservice_DATA = nm-pptp-service.name

BUILT_SOURCES =

###############################################################################

shared_sources = \
	shared/nm-default.h \
	shared/nm-utils/gsystem-local-alloc.h \
	shared/nm-utils/nm-glib.h \
	shared/nm-utils/nm-macros-internal.h \
	shared/nm-utils/nm-shared-utils.c \
	shared/nm-utils/nm-shared-utils.h

EXTRA_DIST += \
	shared/README

src_cppflags = \
	$(GLIB_CFLAGS) \
	$(LIBNM_CFLAGS) \
	-I$(top_srcdir)/shared \
	-I$(builddir)/src \
	-DNM_PPTP_LOCALEDIR=\"$(datadir)/locale\" \
	-DNM_PPTP_STATEDIR=\"$(pptp_statedir)\" \
	-DPLUGINDIR=\"$(PPPD_PLUGIN_DIR)\"

###############################################################################

noinst_LTLIBRARIES += src/libnm-pptp-pppd-service-dbus.la

nodist_src_libnm_pptp_pppd_service_dbus_la_SOURCES = \
	src/nm-pptp-pppd-service-dbus.c \
	src/nm-pptp-pppd-service-dbus.h

src_libnm_pptp_pppd_service_dbus_la_CPPFLAGS = \
	$(src_cppflags) \
	$(filter-out -DGLIB_VERSION_MAX_ALLOWED%,$(AM_CPPFLAGS))

src/nm-pptp-pppd-service-dbus.h: $(top_srcdir)/src/nm-pptp-pppd-service.xml
	$(AM_V_GEN) gdbus-codegen \
		--generate-c-code $(basename $@) \
		--c-namespace NMDBus \
		--interface-prefix org.freedesktop.NetworkManager \
		$<

src/nm-pptp-pppd-service-dbus.c: src/nm-pptp-pppd-service-dbus.h
	@true

BUILT_SOURCES += \
	src/nm-pptp-pppd-service-dbus.h \
	src/nm-pptp-pppd-service-dbus.c

CLEANFILES += \
	src/nm-pptp-pppd-service-dbus.h \
	src/nm-pptp-pppd-service-dbus.c

EXTRA_DIST += \
	src/nm-pptp-pppd-service.xml

###############################################################################

libexec_PROGRAMS += src/nm-pptp-service

src_nm_pptp_service_SOURCES = \
	$(shared_sources) \
	shared/nm-service-defines.h \
	src/nm-pptp-pppd-status.h \
	src/nm-pptp-service.h \
	src/nm-pptp-service.c
src_nm_pptp_service_CPPFLAGS = $(src_cppflags)
src_nm_pptp_service_LDADD = \
	src/libnm-pptp-pppd-service-dbus.la \
	$(GLIB_LIBS) \
	$(LIBNM_LIBS)

###############################################################################

pppd_plugindir = $(PPPD_PLUGIN_DIR)
pppd_plugin_LTLIBRARIES = src/nm-pptp-pppd-plugin.la

src_nm_pptp_pppd_plugin_la_SOURCES = \
	$(shared_sources) \
	src/nm-pptp-pppd-plugin.c \
	src/nm-pptp-pppd-compat.h \
	src/nm-pptp-pppd-status.h
src_nm_pptp_pppd_plugin_la_CPPFLAGS = $(src_cppflags)
src_nm_pptp_pppd_plugin_la_LDFLAGS = \
	-module -avoid-version
src_nm_pptp_pppd_plugin_la_LIBADD = \
	$(GLIB_LIBS) \
	$(LIBNM_LIBS)

###############################################################################

properties/resources.h: properties/gresource.xml
	$(AM_V_GEN) $(GLIB_COMPILE_RESOURCES) $< --target=$@ --sourcedir=$(srcdir)/properties --generate-header --internal

properties/resources.c: properties/gresource.xml $(shell $(GLIB_COMPILE_RESOURCES) --sourcedir=$(srcdir)/properties --generate-dependencies $(srcdir)/properties/gresource.xml)
	$(AM_V_GEN) $(GLIB_COMPILE_RESOURCES) $< --target=$@ --sourcedir=$(srcdir)/properties --generate-source --internal

gtk4/resources.c: properties/gresource.xml $(shell $(GLIB_COMPILE_RESOURCES) --generate-dependencies $(srcdir)/properties/gresource.xml |sed "s,^,$(builddir)/gtk4/,")
	@mkdir -p $(builddir)/gtk4
	$(AM_V_GEN) $(GLIB_COMPILE_RESOURCES) $< --target=$@ --sourcedir=$(srcdir)/gtk4 --sourcedir=$(builddir)/gtk4 --generate-source --internal

gtk4/%.ui: properties/%.ui
	@mkdir -p $(builddir)/gtk4
	gtk4-builder-tool simplify --3to4 $< |grep -v can-default >$@

# Include a prebuilt file in tarball, to avoid hitting
# https://gitlab.gnome.org/GNOME/gtk/-/merge_requests/4415
EXTRA_DIST += \
	gtk4/nm-pptp-dialog.ui

plugin_sources = \
	shared/nm-utils/nm-vpn-plugin-macros.h \
	properties/nm-pptp-editor-plugin.c \
	properties/nm-pptp-editor-plugin.h

editor_sources = \
	$(shared_sources) \
	shared/nm-utils/nm-vpn-plugin-utils.h \
	properties/advanced-dialog.c \
	properties/advanced-dialog.h \
	properties/nm-pptp-editor.c \
	properties/nm-pptp-editor.h

properties_cppflags = \
	-DLOCALEDIR=\"$(datadir)/locale\" \
	-I$(top_srcdir)/shared

CLEANFILES += \
	properties/resources.c \
	properties/resources.h \
	gtk4/resources.c \
	gtk4/nm-pptp-dialog.ui

EXTRA_DIST += \
	properties/nm-pptp-dialog.ui \
	properties/gresource.xml \
	properties/libnm-vpn-plugin-pptp.ver \
	properties/libnm-vpn-plugin-pptp-editor.ver

###############################################################################

plugin_LTLIBRARIES += properties/libnm-vpn-plugin-pptp.la

properties_libnm_vpn_plugin_pptp_la_SOURCES = \
	shared/nm-utils/nm-vpn-plugin-utils.c \
	$(plugin_sources)
properties_libnm_vpn_plugin_pptp_la_CFLAGS = \
	$(GLIB_CFLAGS) \
	$(LIBNM_CFLAGS)
properties_libnm_vpn_plugin_pptp_la_CPPFLAGS = \
	-DNETWORKMANAGER_COMPILATION=NM_NETWORKMANAGER_COMPILATION_LIB_BASE \
	$(properties_cppflags)
properties_libnm_vpn_plugin_pptp_la_LIBADD = \
	$(GLIB_LIBS) \
	$(LIBNM_LIBS)
properties_libnm_vpn_plugin_pptp_la_LDFLAGS = \
	-avoid-version \
	-Wl,--version-script="$(srcdir)/properties/libnm-vpn-plugin-pptp.ver"

###############################################################################

if WITH_GNOME
plugin_LTLIBRARIES += properties/libnm-vpn-plugin-pptp-editor.la
endif

properties_libnm_vpn_plugin_pptp_editor_la_SOURCES = \
	$(editor_sources)
nodist_properties_libnm_vpn_plugin_pptp_editor_la_SOURCES = \
	properties/resources.c \
	properties/resources.h
properties_libnm_vpn_plugin_pptp_editor_la_CFLAGS = \
	$(GLIB_CFLAGS) \
	$(GTK_CFLAGS) \
	$(LIBNM_CFLAGS) \
	$(LIBNMA_CFLAGS)
properties_libnm_vpn_plugin_pptp_editor_la_CPPFLAGS = \
	-DNETWORKMANAGER_COMPILATION=NM_NETWORKMANAGER_COMPILATION_LIB_EDITOR \
	$(properties_cppflags)
properties_libnm_vpn_plugin_pptp_editor_la_LIBADD = \
	$(GLIB_LIBS) \
	$(GTK_LIBS) \
	$(LIBNM_LIBS) \
	$(LIBNMA_LIBS)
properties_libnm_vpn_plugin_pptp_editor_la_LDFLAGS = \
	-avoid-version \
	-Wl,--version-script="$(srcdir)/properties/libnm-vpn-plugin-pptp-editor.ver"

###############################################################################

if WITH_GTK4
plugin_LTLIBRARIES += properties/libnm-gtk4-vpn-plugin-pptp-editor.la
endif

properties_libnm_gtk4_vpn_plugin_pptp_editor_la_SOURCES = \
	$(editor_sources)
nodist_properties_libnm_gtk4_vpn_plugin_pptp_editor_la_SOURCES = \
	gtk4/resources.c \
	properties/resources.h
properties_libnm_gtk4_vpn_plugin_pptp_editor_la_CFLAGS = \
	$(GLIB_CFLAGS) \
	$(GTK4_CFLAGS) \
	$(LIBNM_CFLAGS) \
	$(LIBNMA_GTK4_CFLAGS)
properties_libnm_gtk4_vpn_plugin_pptp_editor_la_CPPFLAGS = \
	$(properties_libnm_vpn_plugin_pptp_editor_la_CPPFLAGS)
properties_libnm_gtk4_vpn_plugin_pptp_editor_la_LIBADD = \
	$(GLIB_LIBS) \
	$(GTK4_LIBS) \
	$(LIBNM_LIBS) \
	$(LIBNMA_GTK4_LIBS)
properties_libnm_gtk4_vpn_plugin_pptp_editor_la_LDFLAGS = \
	$(properties_libnm_vpn_plugin_pptp_editor_la_LDFLAGS)

###############################################################################

if WITH_GNOME
libexec_PROGRAMS += auth-dialog/nm-pptp-auth-dialog
endif

auth_dialog_nm_pptp_auth_dialog_SOURCES = \
	$(shared_sources) \
	auth-dialog/main.c
auth_dialog_nm_pptp_auth_dialog_CFLAGS = \
	$(GLIB_CFLAGS) \
	$(GTK_CFLAGS) \
	$(LIBNM_CFLAGS) \
	$(LIBNMA_CFLAGS) \
	$(LIBSECRET_CFLAGS)
auth_dialog_nm_pptp_auth_dialog_CPPFLAGS = \
	-I$(top_srcdir)/shared
auth_dialog_nm_pptp_auth_dialog_LDADD = \
	$(GLIB_LIBS) \
	$(GTK_LIBS) \
	$(LIBNM_LIBS) \
	$(LIBNMA_LIBS) \
	$(LIBSECRET_LIBS)

###############################################################################

appdatadir = $(datadir)/metainfo
appdata_files = $(appdata_in_files:.xml.in=.xml)
if WITH_GNOME
appdata_DATA = $(appdata_files)
endif
appdata_in_files = appdata/network-manager-pptp.metainfo.xml.in

appdata/network-manager-pptp.metainfo.xml: appdata/network-manager-pptp.metainfo.xml.in
	$(AM_V_GEN)$(MSGFMT) --xml --template $< -d $(top_srcdir)/po -o $@

nm-pptp-service.name: $(srcdir)/nm-pptp-service.name.in
	$(AM_V_GEN) sed -e 's|[@]LIBEXECDIR[@]|$(libexecdir)|g' \
	    -e 's|[@]PLUGINDIR[@]/|@NM_PLUGIN_DIR_NAME_FILE@|g' \
	    $^ >$@

EXTRA_DIST += \
	nm-pptp-service.name.in \
	$(dbusservice_DATA) \
	$(appdata_in_files) \
	$(appdata_files)

CLEANFILES += \
	$(nmvpnservice_DATA) \
	$(appdata_files)

ACLOCAL_AMFLAGS = -I m4
