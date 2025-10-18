plugindir = $(libdir)/NetworkManager
plugin_LTLIBRARIES = libnm-vpn-plugin-iodine.la
if WITH_GNOME
plugin_LTLIBRARIES += libnm-iodine-properties.la
endif

libnm_vpn_plugin_iodine_la_SOURCES =                    \
        nm-iodine.c \
        nm-iodine.h

libnm_iodine_properties_la_SOURCES =                    \
        $(libnm_vpn_plugin_iodine_la_SOURCES)

uidir = $(datadir)/gnome-vpn-properties/iodine
ui_DATA = nm-iodine-dialog.ui

common_CFLAGS =                                         \
        $(GTK_CFLAGS)                                   \
	-I$(top_srcdir)/src				\
        -DICONDIR=\""$(datadir)/pixmaps"\"              \
        -DUIDIR=\""$(uidir)"\"                          \
        -DG_DISABLE_DEPRECATED                          \
        -DGDK_DISABLE_DEPRECATED                        \
        -DGNOME_DISABLE_DEPRECATED                      \
        -DGNOMELOCALEDIR=\"$(datadir)/locale\"		\
        -DVERSION=\"$(VERSION)\"

libnm_vpn_plugin_iodine_la_CFLAGS =                     \
        $(LIBNM_CFLAGS)                                 \
        $(LIBNMA_CFLAGS)                                \
        $(common_CFLAGS)

libnm_iodine_properties_la_CFLAGS =                     \
        $(LIBNM_CFLAGS)                                 \
        $(common_CFLAGS)

libnm_vpn_plugin_iodine_la_LIBADD =                     \
        $(GTK_LIBS)                                     \
        $(LIBNMA_LIBS)                                  \
        $(LIBNM_LIBS)

libnm_iodine_properties_la_LIBADD =                     \
        $(GTK_LIBS)

libnm_vpn_plugin_iodine_la_LDFLAGS =                    \
        -avoid-version

libnm_iodine_properties_la_LDFLAGS =                    \
        $(libnm_vpn_plugin_iodine_la_LDFLAGS)

CLEANFILES = *.bak *~

EXTRA_DIST =                            \
        $(ui_DATA)
