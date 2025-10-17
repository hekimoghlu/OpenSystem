/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 20, 2023.
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
#include "krb5_locl.h"


KRB5_LIB_VARIABLE const char *krb5_config_file =
#ifdef __APPLE_TARGET_EMBEDDED__
"%{ApplicationResources}/com.apple.Kerberos.plist" PATH_SEP
"~/Library/Preferences/com.apple.Kerberos.plist" PATH_SEP
"~/Library/Preferences/edu.mit.Kerberos" PATH_SEP
"~/.krb5/config" PATH_SEP
#else /* __APPLE_TARGET_EMBEDDED__ */
#ifdef __APPLE__
"%{ApplicationResources}/com.apple.Kerberos.plist" PATH_SEP
"~/Library/Preferences/com.apple.Kerberos.plist" PATH_SEP
"/Library/Preferences/com.apple.Kerberos.plist" PATH_SEP
"~/Library/Preferences/edu.mit.Kerberos" PATH_SEP
"/Library/Preferences/edu.mit.Kerberos" PATH_SEP
#endif	/* __APPLE__ */
"~/.krb5/config" PATH_SEP
SYSCONFDIR "/krb5.conf"
#ifdef _WIN32
PATH_SEP "%{COMMON_APPDATA}/Kerberos/krb5.conf"
PATH_SEP "%{WINDOWS}/krb5.ini"
#else /* _WIN32 */
PATH_SEP "/etc/krb5.conf"
#endif /* _WIN32 */
#endif /* __APPLE_TARGET_EMBEDDED__ */
;

KRB5_LIB_VARIABLE const char *krb5_defkeyname = KEYTAB_DEFAULT;

KRB5_LIB_VARIABLE const char *krb5_cc_type_api = "API";
KRB5_LIB_VARIABLE const char *krb5_cc_type_file = "FILE";
KRB5_LIB_VARIABLE const char *krb5_cc_type_memory = "MEMORY";
KRB5_LIB_VARIABLE const char *krb5_cc_type_kcm = "KCM";
KRB5_LIB_VARIABLE const char *krb5_cc_type_scc = "SCC";
KRB5_LIB_VARIABLE const char *krb5_cc_type_kcc = "KCC";
