/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 28, 2024.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#include <sys/cdefs.h>

__FBSDID("$FreeBSD$");

#ifndef lint
static const char copyright[] =
"@(#) Copyright (c) 1993\n\
	The Regents of the University of California.  All rights reserved.\n";
#endif

#ifndef lint
static const char sccsid[] = "@(#)uname.c	8.2 (Berkeley) 5/4/95";
#endif

#include <sys/param.h>
#include <sys/sysctl.h>

#include <err.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#ifdef __APPLE__
#include <string.h>
#include <get_compat.h>
#else
#include <osreldate.h>

#define	COMPAT_MODE(a,b)	(1)
#endif

#define	MFLAG	0x01
#define	NFLAG	0x02
#define	PFLAG	0x04
#define	RFLAG	0x08
#define	SFLAG	0x10
#define	VFLAG	0x20
#ifndef __APPLE__
#define	IFLAG	0x40
#define	UFLAG	0x80
#define	KFLAG	0x100
#define	BFLAG	0x200
#endif

typedef void (*get_t)(void);
#ifdef __APPLE__
static get_t get_platform, get_hostname, get_arch,
    get_release, get_sysname, get_version;
#else
static get_t get_buildid, get_ident, get_platform, get_hostname, get_arch,
    get_release, get_sysname, get_kernvers, get_uservers, get_version;
#endif

#ifndef __APPLE__
static void native_ident(void);
#endif
static void native_platform(void);
static void native_hostname(void);
static void native_arch(void);
static void native_release(void);
static void native_sysname(void);
static void native_version(void);
#ifndef __APPLE__
static void native_kernvers(void);
static void native_uservers(void);
static void native_buildid(void);
#endif
static void print_uname(u_int);
static void setup_get(void);
static void usage(void);

#ifdef __APPLE__
static char *platform, *hostname, *arch, *release, *sysname,
    *version;
#else
static char *buildid, *ident, *platform, *hostname, *arch, *release, *sysname,
    *version, *kernvers, *uservers;
#endif
static int space;

int
main(int argc, char *argv[])
{
	u_int flags;
	int ch;

	setup_get();
	flags = 0;

#ifdef __APPLE__
	while ((ch = getopt(argc, argv, "amnoprsv")) != -1)
#else
	while ((ch = getopt(argc, argv, "abiKmnoprsUv")) != -1)
#endif
		switch(ch) {
		case 'a':
			flags |= (MFLAG | NFLAG | RFLAG | SFLAG | VFLAG);
#ifdef __APPLE__
			if (!COMPAT_MODE("bin/uname", "Unix2003")) {
				flags |= PFLAG;
			}
#endif
			break;
#ifndef __APPLE__
		case 'b':
			flags |= BFLAG;
			break;
		case 'i':
			flags |= IFLAG;
			break;
		case 'K':
			flags |= KFLAG;
			break;
#endif
		case 'm':
			flags |= MFLAG;
			break;
		case 'n':
			flags |= NFLAG;
			break;
		case 'p':
			flags |= PFLAG;
			break;
		case 'r':
			flags |= RFLAG;
			break;
		case 's':
		case 'o':
			flags |= SFLAG;
			break;
#ifndef __APPLE__
		case 'U':
			flags |= UFLAG;
			break;
#endif
		case 'v':
			flags |= VFLAG;
			break;
		case '?':
		default:
			usage();
		}

	argc -= optind;
	argv += optind;

	if (argc)
		usage();

	if (!flags)
		flags |= SFLAG;

	print_uname(flags);
#ifdef __APPLE__
	if (ferror(stdout) != 0 || fflush(stdout) != 0)
		err(1, "stdout");
#endif
	exit(0);
}

#ifdef __APPLE__

#ifndef nitems
#define	nitems(x)	(sizeof((x)) / sizeof((x)[0]))
#endif

static const struct env_optmap {
	const char *shortopt;
	const char *envopt;
} env_opts[] = {
	{ "m",	"UNAME_MACHINE" },
	{ "n",	"UNAME_NODENAME" },
	{ "r",	"UNAME_RELEASE" },
	{ "s",	"UNAME_SYSNAME" },
	{ "v",	"UNAME_VERSION" },
};

/*
 * Scan the environment for an override.  upstream_optalias is the 1:1 name that
 * upstream uses (e.g., -n => UNAME_n); we'll fallback to it if the mapped
 * long option does not exist.  Our aliases take precedence.
 */
static inline char *
scan_env(const char *opt, const char *upstream_optalias)
{
	const struct env_optmap *om;
	char *ret;
	size_t i;

	for (i = 0; i < nitems(env_opts); i++) {
		om = &env_opts[i];
		if (strcmp(om->shortopt, opt) == 0) {
			if ((ret = getenv(om->envopt)) != NULL)
				return (ret);

			break;
		}
	}

	return (getenv(upstream_optalias));
}

#define	CHECK_ENV(opt,var)					\
do {								\
	if ((var = scan_env(opt, "UNAME_" opt)) == NULL) {	\
		get_##var = native_##var;			\
	} else {						\
		get_##var = (get_t)NULL;			\
	}							\
} while (0)

#else	/* !__APPLE__ */

#define	CHECK_ENV(opt,var)				\
do {							\
	if ((var = getenv("UNAME_" opt)) == NULL) {	\
		get_##var = native_##var;		\
	} else {					\
		get_##var = (get_t)NULL;		\
	}						\
} while (0)

#endif	/* __APPLE__ */

static void
setup_get(void)
{
	CHECK_ENV("s", sysname);
	CHECK_ENV("n", hostname);
	CHECK_ENV("r", release);
	CHECK_ENV("v", version);
	CHECK_ENV("m", platform);
	CHECK_ENV("p", arch);
#ifndef __APPLE__
	CHECK_ENV("i", ident);
	CHECK_ENV("K", kernvers);
	CHECK_ENV("U", uservers);
	CHECK_ENV("b", buildid);
#endif
}

#define	PRINT_FLAG(flags,flag,var)		\
	if ((flags & flag) == flag) {		\
		if (space)			\
			printf(" ");		\
		else				\
			space++;		\
		if (get_##var != NULL)		\
			(*get_##var)();		\
		printf("%s", var);		\
	}

static void
print_uname(u_int flags)
{
	PRINT_FLAG(flags, SFLAG, sysname);
	PRINT_FLAG(flags, NFLAG, hostname);
	PRINT_FLAG(flags, RFLAG, release);
	PRINT_FLAG(flags, VFLAG, version);
	PRINT_FLAG(flags, MFLAG, platform);
	PRINT_FLAG(flags, PFLAG, arch);
#ifndef __APPLE__
	PRINT_FLAG(flags, IFLAG, ident);
	PRINT_FLAG(flags, KFLAG, kernvers);
	PRINT_FLAG(flags, UFLAG, uservers);
	PRINT_FLAG(flags, BFLAG, buildid);
#endif
	printf("\n");
}

#define	NATIVE_SYSCTL2_GET(var,mib0,mib1)	\
static void					\
native_##var(void)				\
{						\
	int mib[] = { (mib0), (mib1) };		\
	size_t len;				\
	static char buf[1024];			\
	char **varp = &(var);			\
						\
	len = sizeof buf;			\
	if (sysctl(mib, sizeof mib / sizeof mib[0],	\
	   &buf, &len, NULL, 0) == -1)		\
		err(1, "sysctl");

#define	NATIVE_SYSCTLNAME_GET(var,name)		\
static void					\
native_##var(void)				\
{						\
	size_t len;				\
	static char buf[1024];			\
	char **varp = &(var);			\
						\
	len = sizeof buf;			\
	if (sysctlbyname(name, &buf, &len, NULL,\
	    0) == -1)				\
		err(1, "sysctlbyname");

#define	NATIVE_SET				\
	*varp = buf;				\
	return;					\
}	struct __hack

#define	NATIVE_BUFFER	(buf)
#define	NATIVE_LENGTH	(len)

NATIVE_SYSCTL2_GET(sysname, CTL_KERN, KERN_OSTYPE) {
} NATIVE_SET;

NATIVE_SYSCTL2_GET(hostname, CTL_KERN, KERN_HOSTNAME) {
} NATIVE_SET;

NATIVE_SYSCTL2_GET(release, CTL_KERN, KERN_OSRELEASE) {
} NATIVE_SET;

NATIVE_SYSCTL2_GET(version, CTL_KERN, KERN_VERSION) {
	size_t n;
	char *p;

	p = NATIVE_BUFFER;
	n = NATIVE_LENGTH;
	for (; n--; ++p)
		if (*p == '\n' || *p == '\t')
			*p = ' ';
} NATIVE_SET;

NATIVE_SYSCTL2_GET(platform, CTL_HW, HW_MACHINE) {
} NATIVE_SET;

#ifdef __APPLE__

static void
native_arch(void)
{

#if defined(__ppc__) || defined(__ppc64__)
	arch = "powerpc";
#elif defined(__i386__) || defined(__x86_64__)
	arch = "i386";
#elif defined(__arm__) || defined(__arm64__)
	arch = "arm";
#else
	arch = "unknown";
#endif
}

#else

NATIVE_SYSCTL2_GET(arch, CTL_HW, HW_MACHINE_ARCH) {
} NATIVE_SET;

NATIVE_SYSCTLNAME_GET(ident, "kern.ident") {
} NATIVE_SET;

NATIVE_SYSCTLNAME_GET(buildid, "kern.build_id") {
} NATIVE_SET;

static void
native_uservers(void)
{
	static char buf[128];

	snprintf(buf, sizeof(buf), "%d", __FreeBSD_version);
	uservers = buf;
}

static void
native_kernvers(void)
{
	static char buf[128];

	snprintf(buf, sizeof(buf), "%d", getosreldate());
	kernvers = buf;
}
#endif

static void
usage(void)
{
#ifdef __APPLE__
	fprintf(stderr, "usage: uname [-amnoprsv]\n");
#else
	fprintf(stderr, "usage: uname [-abiKmnoprsUv]\n");
#endif
	exit(1);
}
