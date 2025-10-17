/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 23, 2024.
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
#ifndef _NETSMB_SMB_LIB_H_
#define _NETSMB_SMB_LIB_H_

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <CoreFoundation/CoreFoundation.h>
#include <pthread.h>
#include <sys/mount.h>

#include <netsmb/smb.h>
#include <netsmb/smb_dev.h>
#include <netsmb/netbios.h>
#include <os/log.h>
#include "preference.h"

#define SMB_BonjourServiceNameType "_smb._tcp."
#define NetBIOS_SMBSERVER		"*SMBSERVER"

/* Used by mount_smbfs to pass mount option into the smb_mount call */
#define kNotifyOffMountKey	CFSTR("SMBNotifyOffMount")
#define kStreamstMountKey	CFSTR("SMBStreamsMount")
#define kdirModeKey			CFSTR("SMBDirModes")
#define kfileModeKey		CFSTR("SMBFileModes")
#define kHighFidelityMountKey CFSTR("SMBHighFidelityMount") /* Also an Open option! */
#define kDataCacheOffMountKey CFSTR("SMBDataCacheOffMount")
#define kMDataCacheOffMountKey CFSTR("SMBMDataCacheOffMount")
#define kSnapshotTimeKey CFSTR("SMBSnapshotTimeMount")
#define kSessionEncryptionKey CFSTR("SMBSessionEncryption") /* Also an Open option! */
#define kShareEncryptionKey CFSTR("SMBShareEncryption")

/* This one should be defined in NetFS.h, but we want it to be private */
#define kTimeMachineMountKey CFSTR("TimeMachineMount")

#define SMB_PASSWORD_KEY "Password"

struct connectAddress {
	int	so;
	union {
		struct sockaddr addr;
		struct sockaddr_in in4;
		struct sockaddr_in6 in6;
		struct sockaddr_nb nb;
		struct sockaddr_storage storage;	/* Make we always have enough room */
	};
};

/*
 * nb environment
 */
struct nb_ctx {
	struct sockaddr_in	nb_ns;			/* ip addr of name server */
	struct sockaddr_in	nb_sender;		/* address of the system that responded */
};

/*
 * SMB work context. Used to store all values which are necessary
 * to establish connection to an SMB server.
 */
struct smb_ctx {
	pthread_mutex_t ctx_mutex;
	CFURLRef		ct_url;
	uint32_t		ct_flags;	/* SMBCF_ */
	int				ct_fd;		/* handle of connection */
	uint16_t		ct_cancel;
	CFStringRef		serverNameRef; /* Display Server name obtain from URL or Bonjour Service Name */
	char *			serverName;		/* Server name obtain from the URL */
	struct nb_ctx		ct_nb;
	struct smbioc_ossn	ct_ssn;
	struct smbioc_setup ct_setup;
	struct smbioc_share	ct_sh;
	struct sockaddr	*ct_saddr;
	char *			ct_origshare;
	CFStringRef		mountPath;
	uint32_t		ct_session_uid;
	uint32_t		ct_session_caps;		/* Obtained from the negotiate message */
	uint32_t		ct_session_smb2_caps;
	uint32_t		ct_session_flags;	/* Obtained from the negotiate message */
    uint64_t		ct_session_misc_flags;
	uint32_t		ct_session_hflags;
	uint32_t		ct_session_hflags2;
	uint32_t		ct_session_shared;	/* Obtained from the negotiate message, currently only tells if the session is shared */
	uint64_t		ct_session_txmax;				
	uint64_t		ct_session_rxmax;
    uint64_t		ct_session_wxmax;
	int				forceNewSession;
	int				inCallback;
	int				serverIsDomainController;
	CFDictionaryRef mechDict;
	struct smb_prefs prefs;
    char *          model_info;     /* SMB 2/3 Server model string, only MAC to MAC */
};

/* smb_ctx ct_flags */
#define	SMBCF_RESOLVED			0x00000001	/* We have resolved the address and name */
#define	SMBCF_CONNECTED			0x00000002	/* The negotiate message was succesful */
#define	SMBCF_AUTHORIZED		0x00000004	/* We have completed the security phase */
#define	SMBCF_SHARE_CONN		0x00000008	/* We have a tree connection */
#define	SMBCF_READ_PREFS		0x00000010	/* We already read the preferences */
#define SMBCF_RAW_NTLMSSP		0x00000040	/* Server only supports RAW NTLMSSP */
#define SMBCF_MATCH_DNS         0x00000080	/* Check for already mounted servers using dns name */
#define SMBCF_FORCE_NEW_SESSION 0x00000100  /* Force New Session was used during the Open Session */
#define SMBCF_HIFI_REQUESTED    0x00000200  /* HiFi mode is being requested */
#define SMBCF_SESSION_ENCRYPT   0x00000400  /* Force session encryption */
#define SMBCF_SHARE_ENCRYPT     0x00000800  /* Force share encryption */
#define SMBCF_EXPLICITPWD		0x00010000	/* The password was set by the url */

#define SMBCF_CONNECT_STATE	SMBCF_CONNECTED | SMBCF_AUTHORIZED | SMBCF_SHARE_CONN

__BEGIN_DECLS

struct sockaddr;

int smb_load_library(void);

void smb_ctx_hexdump(const char */* func */, const char */* comments */, unsigned char */* buf */, size_t /* inlen */);

/*
 * Context management
 */
CFMutableDictionaryRef CreateAuthDictionary(struct smb_ctx *ctx, uint32_t authFlags,
											const char * clientPrincipal, 
											uint32_t clientNameType);
int smb_ctx_clone(struct smb_ctx *new_ctx, struct smb_ctx *old_ctx,
				  CFMutableDictionaryRef openOptions);
int findMountPointSession(void *inRef, const char *mntPoint);
void *create_smb_ctx(void);
int  create_smb_ctx_with_url(struct smb_ctx **out_ctx, const char *url);
int smb_ctx_gethandle(struct smb_ctx *ctx);
void smb_ctx_cancel_connection(struct smb_ctx *ctx);
void smb_ctx_done(void *);
int already_mounted(struct smb_ctx *ctx, char *uppercaseShareName, struct statfs *fs, 
					int fs_cnt, CFMutableDictionaryRef mdict, int requestMntFlags);

Boolean SMBGetDictBooleanValue(CFDictionaryRef Dict, const void * KeyValue, Boolean DefaultValue);

int smb_get_server_info(struct smb_ctx *ctx, CFURLRef url, CFDictionaryRef OpenOptions, CFDictionaryRef *ServerParams);
int smb_open_session(struct smb_ctx *ctx, CFURLRef url, CFDictionaryRef OpenOptions, CFDictionaryRef *sessionInfo);
int smb_mount(struct smb_ctx *in_ctx, CFStringRef mpoint, 
			  CFDictionaryRef mOptions, CFDictionaryRef *mInfo,
			  void (*)(void  *, void *), void *);

void smb_get_session_properties(struct smb_ctx *ctx);
int smb_share_disconnect(struct smb_ctx *ctx);
int smb_share_connect(struct smb_ctx *ctx);
uint16_t smb_tree_conn_optional_support_flags(struct smb_ctx *ctx);
uint32_t smb_tree_conn_fstype(struct smb_ctx *ctx);
int  smb_ctx_setuser(struct smb_ctx *, const char *);
int  smb_ctx_setshare(struct smb_ctx *, const char *);
int  smb_ctx_setdomain(struct smb_ctx *, const char *);
int  smb_ctx_setpassword(struct smb_ctx *, const char *, int /*setFlags*/);

uint16_t smb_ctx_connstate(struct smb_ctx *ctx);
int  smb_smb_open_print_file(struct smb_ctx *, int, int, const char *, smbfh*);
int  smb_smb_close_print_file(struct smb_ctx *, smbfh);
int  smb_read(struct smb_ctx *, smbfh, off_t, uint32_t, char *);
int  smb_write(struct smb_ctx *, smbfh, off_t, uint32_t, const char *);
void smb_ctx_get_user_mount_info(const char * /*mntonname */, CFMutableDictionaryRef);

CF_RETURNS_RETAINED CFArrayRef smb_resolve_domain(CFStringRef serverNameRef);

__END_DECLS

#endif /* _NETSMB_SMB_LIB_H_ */
