AM_CPPFLAGS = \
	$(LIBNM_CFLAGS) \
	-DG_DISABLE_DEPRECATED \
	-DBINDIR=\"$(bindir)\" \
	-DPREFIX=\""$(prefix)"\" \
	-DSYSCONFDIR=\""$(sysconfdir)"\" \
	-DVERSION="\"$(VERSION)\"" \
	-DLIBDIR=\""$(libdir)"\" \
	-DLIBEXECDIR=\""$(libexecdir)"\" \
	-DLOCALSTATEDIR=\""$(localstatedir)"\" \
	-DDATADIR=\"$(datadir)\"

libexec_PROGRAMS = \
	nm-iodine-service

nm_iodine_service_SOURCES = \
	nm-iodine-service.c \
	nm-iodine-service.h \
	nm-iodine-service-defines.h

nm_iodine_service_LDADD = \
	$(LIBNM_LIBS)

CLEANFILES = *~
