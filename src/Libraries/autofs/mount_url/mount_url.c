/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 24, 2023.
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
#include <unistd.h>
#include <mntopts.h>
#include <syslog.h>

#include <sys/mount.h>

#include <CoreFoundation/CoreFoundation.h>
#include <NetFS/NetFS.h>
#include <NetFS/NetFSPrivate.h>
/*
 * XXX - <NetAuth/NetAuth.h> include <NetAuth/NAKeys.h>, which redefines
 * some of our #defines if _NETFS_H_ isn't defined.
 *
 * To prevent that from happening, we include it *after* including NetFS.h,
 * which defines _NETFS_H_.
 */
#include <NetAuth/NetAuth.h>

#define ALT_SOFT        0x00000001

static const struct mntopt mopts_std[] = {
	MOPT_STDOPTS,
	MOPT_UPDATE,
	MOPT_RELOAD,
	{ "soft", 0, ALT_SOFT, 1 },
	{ NULL, 0, 0, 0 }
};

static void usage(void);

static int do_mount_direct(CFURLRef server_URL, CFStringRef mountdir,
    CFDictionaryRef open_options, CFDictionaryRef mount_options,
    CFDictionaryRef *mount_infop);

int
main(int argc, char **argv)
{
	int c;
	int usenetauth = 0;
	mntoptparse_t mp;
	int flags, altflags;
	CFURLRef URL;
	CFStringRef mountdir_CFString;
	CFMutableDictionaryRef open_options, mount_options;
	CFDictionaryRef mount_info;
	CFNumberRef mount_options_flags;
	int res;

	flags = altflags = 0;
	getmnt_silent = 1;
	while ((c = getopt(argc, argv, "no:rw")) != -1) {
		switch (c) {
		case 'n':
			usenetauth = 1;
			break;

		case 'o':
			/*
			 * OK, parse these options, and update the flags.
			 */
			mp = getmntopts(optarg, mopts_std, &flags, &altflags);
			freemntopts(mp);
			break;

		case 'r':
			flags |= MNT_RDONLY;
			break;

		case 'w':
			flags &= ~MNT_RDONLY;
			break;

		case '?':
		default:
			usage();
			break;
		}
	}
	argc -= optind;
	argv += optind;

	if (argc != 2) {
		usage();
	}

	/*
	 * Nothing can stop the Duke of...
	 */
	URL = CFURLCreateWithBytes(kCFAllocatorDefault, (const UInt8 *)argv[0],
	    strlen(argv[0]), kCFStringEncodingUTF8, NULL);
	if (URL == NULL) {
		exit(ENOMEM);
	}

	mountdir_CFString = CFStringCreateWithCString(kCFAllocatorDefault,
	    argv[1], kCFStringEncodingUTF8);
	if (mountdir_CFString == NULL) {
		exit(ENOMEM);
	}

	open_options = CFDictionaryCreateMutable(kCFAllocatorDefault, 0,
	    &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
	if (open_options == NULL) {
		exit(ENOMEM);
	}
	/*
	 * It's OK to use an existing session.
	 */
	CFDictionaryAddValue(open_options, kNetFSForceNewSessionKey,
	    kCFBooleanFalse);
	/*
	 * And it's OK to mount something from ourselves.
	 */
	CFDictionaryAddValue(open_options, kNetFSAllowLoopbackKey,
	    kCFBooleanTrue);
	/*
	 * This could be mounting a home directory, so we don't want
	 * the mount to look at user preferences in the home directory.
	 */
	CFDictionaryAddValue(open_options, kNetFSNoUserPreferencesKey,
	    kCFBooleanTrue);
	/*
	 * We don't want any UI popped up for the mount.
	 */
	CFDictionaryAddValue(open_options, kUIOptionKey, kUIOptionNoUI);

	mount_options = CFDictionaryCreateMutable(kCFAllocatorDefault, 0,
	    &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
	if (mount_options == NULL) {
		exit(ENOMEM);
	}
	/*
	 * It's OK to use an existing session.
	 */
	CFDictionaryAddValue(mount_options, kNetFSForceNewSessionKey,
	    kCFBooleanFalse);
	/*
	 * We want the URL mounted exactly where we specify.
	 */
	CFDictionaryAddValue(mount_options, kNetFSMountAtMountDirKey,
	    kCFBooleanTrue);
	/*
	 * This could be mounting a home directory, so we don't want
	 * the mount to look at user preferences in the home directory.
	 */
	CFDictionaryAddValue(mount_options, kNetFSNoUserPreferencesKey,
	    kCFBooleanTrue);
	/*
	 * We want to allow the URL to specify a directory underneath
	 * a share point for file systems that support the notion of
	 * shares.
	 */
	CFDictionaryAddValue(mount_options, kNetFSAllowSubMountsKey,
	    kCFBooleanTrue);
	/*
	 * Add the mount flags.
	 */
	mount_options_flags = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &flags);
	CFDictionaryAddValue(mount_options, kNetFSMountFlagsKey, mount_options_flags);
	CFRelease(mount_options_flags);
	/*
	 * Add the soft mount flag.
	 */
	CFDictionaryAddValue(mount_options, kNetFSSoftMountKey,
	    (altflags & ALT_SOFT) ? kCFBooleanTrue : kCFBooleanFalse);
	/*
	 * We don't want any UI popped up for the mount.
	 */
	CFDictionaryAddValue(mount_options, kUIOptionKey, kUIOptionNoUI);

	if (usenetauth) {
		res = NAConnectToServerSync(URL, mountdir_CFString,
		    open_options, mount_options, &mount_info);
	} else {
		res = do_mount_direct(URL, mountdir_CFString, open_options,
		    mount_options, &mount_info);
	}
	/*
	 * 0 means "no error", EEXIST means "that's already mounted, and
	 * mountinfo says where it's mounted".  In those cases, a
	 * directory of mount information was returned; release it.
	 */
	if ((res == 0 || res == EEXIST) && (mount_info != NULL)) {
		CFRelease(mount_info);
	}
	CFRelease(mount_options);
	CFRelease(open_options);
	CFRelease(mountdir_CFString);
	CFRelease(URL);
	if (res != 0) {
		/*
		 * Report any failure status that doesn't fit in the
		 * 8 bits of a UN*X exit status, and map it to EIO
		 * by default and EAUTH for ENETFS errors.
		 */
		if ((res & 0xFFFFFF00) != 0) {
			syslog(LOG_ERR,
			    "mount_url: Mount of %1024s on %1024s gives status %d",
			    argv[0], argv[1], res);

			switch (res) {
			case ENETFSACCOUNTRESTRICTED:
			case ENETFSPWDNEEDSCHANGE:
			case ENETFSPWDPOLICY:
				res = EAUTH;
				break;

			default:
				res = EIO;
				break;
			}
		}
	}

	return res;
}

static void
usage(void)
{
	fprintf(stderr, "Usage: mount_url [-n] [-rw] [-o options] url node\n");
	exit(1);
}

static int
do_mount_direct(CFURLRef server_URL, CFStringRef mountdir,
    CFDictionaryRef open_options, CFDictionaryRef mount_options,
    CFDictionaryRef *mount_infop)
{
	int ret;
	void *session_ref;

	*mount_infop = NULL;
	ret = netfs_CreateSessionRef(server_URL, &session_ref);
	if (ret != 0) {
		return ret;
	}
	ret = netfs_OpenSession(server_URL, session_ref, open_options, NULL);
	if (ret != 0) {
		netfs_CloseSession(session_ref);
		return ret;
	}
	ret = netfs_Mount(session_ref, server_URL, mountdir, mount_options,
	    mount_infop);
	netfs_CloseSession(session_ref);
	return ret;
}
