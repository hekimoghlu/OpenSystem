AC_PREREQ(2.52)

AC_INIT(NetworkManager-iodine, 1.2.1, agx@sigxcpu.org, NetworkManager-iodine)
AM_INIT_AUTOMAKE([subdir-objects no-dist-gzip dist-xz])
AM_MAINTAINER_MODE

AC_CONFIG_MACRO_DIR([m4])

AC_CONFIG_HEADERS([config.h])

dnl
dnl Require programs
dnl
AC_PROG_CC
AM_PROG_CC_C_O
AC_PROG_INSTALL
LT_INIT

dnl
dnl Required headers
dnl
AC_CHECK_HEADERS(fcntl.h paths.h sys/ioctl.h sys/time.h syslog.h unistd.h)

dnl
dnl Checks for typedefs, structures, and compiler characteristics.
dnl
AC_TYPE_MODE_T
AC_TYPE_PID_T

dnl
dnl Checks for library functions.
dnl
AC_PROG_GCC_TRADITIONAL
AC_FUNC_MEMCMP
AC_CHECK_FUNCS(select socket uname)

dnl
dnl GNOME support
dnl
AC_ARG_WITH(gnome, AS_HELP_STRING([--without-gnome], [Build NetworkManager-iodine without GNOME support, e.g. vpn service only]))
AM_CONDITIONAL(WITH_GNOME, test x"$with_gnome" != xno)

AC_ARG_WITH(authdlg, AS_HELP_STRING([--without-authdlg], [Build NetworkManager-iodine without authentication dialog]))
AM_CONDITIONAL(WITH_AUTHDLG, test x"$with_authdlg" != xno)

AC_ARG_ENABLE(absolute-paths, AS_HELP_STRING([--enable-absolute-paths], [Use absolute paths to in .name files. Useful for development. (default is no)]))

GETTEXT_PACKAGE=NetworkManager-iodine
AC_SUBST(GETTEXT_PACKAGE)
AC_DEFINE_UNQUOTED(GETTEXT_PACKAGE,"$GETTEXT_PACKAGE", [Gettext package])

IT_PROG_INTLTOOL([0.35])
AM_GLIB_GNU_GETTEXT

if test x"$with_gnome" != xno; then
	with_gnome=yes
        PKG_CHECK_MODULES(GTK, gtk+-3.0 >= 3.4)
        GTK_CFLAGS="$GTK_CFLAGS -DGDK_VERSION_MIN_REQUIRED=GDK_VERSION_3_4"

        PKG_CHECK_MODULES(LIBNMA, libnma >= 1.7.0)
        PKG_CHECK_MODULES(LIBSECRET, libsecret-1 >= 0.18)
fi

PKG_CHECK_MODULES(LIBNM, libnm >= 1.1.0)
LIBNM_CFLAGS="$LIBNM_CFLAGS -DNM_VERSION_MIN_REQUIRED=NM_VERSION_1_2"
LIBNM_CFLAGS="$LIBNM_CFLAGS -DNM_VERSION_MAX_ALLOWED=NM_VERSION_1_2"

NM_VPN_SERVICE_DIR=`$PKG_CONFIG --define-variable prefix='\${prefix}' --variable vpnservicedir libnm`
AC_SUBST(NM_VPN_SERVICE_DIR)

test x"$enable_absolute_paths" = x"yes" && NM_PLUGIN_DIR='$(libdir)/NetworkManager/'
AC_SUBST(NM_PLUGIN_DIR)

NM_COMPILER_WARNINGS

AC_CONFIG_FILES([
Makefile
src/Makefile
auth-dialog/Makefile
properties/Makefile
po/Makefile.in
])
AC_OUTPUT
