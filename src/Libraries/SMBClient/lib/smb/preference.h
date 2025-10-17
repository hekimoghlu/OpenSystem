/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 5, 2023.
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
#ifndef _SMBLIB_PREFERENCES_H_
#define _SMBLIB_PREFERENCES_H_

#include <netsmb/smb_conn.h>

#define DefaultNetBIOSResolverTimeout	1

/* Shouldn't this be handle by gss */
enum smb_min_auth {
	SMB_MINAUTH = 0,			/* minimum auth level for connection */
	SMB_MINAUTH_LM = 1,			/* No plaintext passwords */
	SMB_MINAUTH_NTLM = 2,		/* don't send LM reply? */
	SMB_MINAUTH_NTLMV2 = 3,		/* don't fall back to NTLMv1 */
	SMB_MINAUTH_KERBEROS = 4	/* don't do NTLMv1 or NTLMv2 */
};

struct smb_prefs {
    CFStringRef         LocalNetBIOSName;
    CFArrayRef          WINSAddresses;
    int32_t             NetBIOSResolverTimeout;
    CFStringEncoding    WinCodePage;
    uint32_t            tryBothPorts;
    uint16_t            tcp_port;
    int32_t             KernelLogLevel;
    enum smb_min_auth   minAuthAllowed;
    int64_t             altflags;
    CFStringRef         NetBIOSDNSName;
    int32_t             protocol_version_map;
    uint32_t            lanman_on;
    uint32_t            signing_required;
    int32_t             signing_algorithm_map;
    int32_t             signing_req_versions;
    int32_t             max_resp_timeout;
    int32_t             ip_QoS;

    int32_t             dir_cache_async_cnt;
    int32_t             dir_cache_max;
    int32_t             dir_cache_min;
    int32_t             max_dirs_cached;
    int32_t             max_dir_entries_cached;

    uint32_t            no_DNS_match;
    uint32_t            try_netBIOS_before_DNS;

    int32_t             read_size[3];
    int32_t             read_count[3];
    int32_t             write_size[3];
    int32_t             write_count[3];
    
    int32_t             rw_max_check_time;
    int32_t             rw_gb_threshold;

    uint32_t            mc_max_channels;
    uint32_t            mc_srvr_rss_channels;
    uint32_t            mc_clnt_rss_channels;
    uint32_t            mc_client_if_ignorelist[kClientIfIgnorelistMaxLen];
    uint32_t            mc_client_if_ignorelist_len;
    
    int32_t             encrypt_algorithm_map;
    uint32_t            force_sess_encrypt;
    uint32_t            force_share_encrypt;

    int32_t             compression_algorithms_map;
    int32_t             compression_io_threshold;
    int32_t             compression_chunk_len;
    int32_t             compression_max_fail_cnt;
    char *              compression_exclude[kClientCompressMaxEntries];
    uint32_t            compression_exclude_cnt;
    char *              compression_include[kClientCompressMaxEntries];
    uint32_t            compression_include_cnt;
};

void getDefaultPreferences(struct smb_prefs *prefs);
void setWINSAddress(struct smb_prefs *prefs, const char *winsAddress, int count);
void releasePreferenceInfo(struct smb_prefs *prefs);
void readPreferences(struct smb_prefs *prefs, char *serverName, char *shareName, 
					 int noUserPrefs, int resetPrefs);
CFStringEncoding getPrefsCodePage( void );

#endif // _SMBLIB_PREFERENCES_H_
