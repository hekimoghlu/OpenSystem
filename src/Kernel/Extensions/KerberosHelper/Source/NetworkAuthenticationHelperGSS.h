/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 15, 2024.
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


#import <KerberosHelper/NetworkAuthenticationHelper.h>
#import <GSS/gssapi.h>

/*
 * GSS-API Support
 */

gss_cred_id_t
NAHSelectionGetGSSCredential(NAHSelectionRef client, CFErrorRef *error);

gss_name_t
NAHSelectionGetGSSAcceptorName(NAHSelectionRef selection, CFErrorRef *error);

gss_OID
NAHSelectionGetGSSMech(NAHSelectionRef client);

/*
 * Turn the AuthenticationInfo dict into something useful
 */

gss_cred_id_t
NAHAuthenticationInfoCopyClientCredential(CFDictionaryRef authInfo, CFErrorRef *error);

gss_name_t
NAHAuthenticationInfoCopyServerName(CFDictionaryRef authInfo, CFErrorRef *error);

gss_OID
NAHAuthenticationInfoGetGSSMechanism(CFDictionaryRef authInfo, CFErrorRef *error);

