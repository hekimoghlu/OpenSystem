/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 6, 2021.
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
#ifndef WKSecurityOriginRef_h
#define WKSecurityOriginRef_h

#include <WebKit/WKBase.h>

#ifdef __cplusplus
extern "C" {
#endif

WK_EXPORT WKTypeID WKSecurityOriginGetTypeID(void);

WK_EXPORT WKSecurityOriginRef WKSecurityOriginCreateFromString(WKStringRef string);
WK_EXPORT WKSecurityOriginRef WKSecurityOriginCreateFromDatabaseIdentifier(WKStringRef identifier);
WK_EXPORT WKSecurityOriginRef WKSecurityOriginCreate(WKStringRef protocol, WKStringRef host, int port);

WK_EXPORT WKStringRef WKSecurityOriginCopyDatabaseIdentifier(WKSecurityOriginRef securityOrigin);
WK_EXPORT WKStringRef WKSecurityOriginCopyToString(WKSecurityOriginRef securityOrigin);
WK_EXPORT WKStringRef WKSecurityOriginCopyProtocol(WKSecurityOriginRef securityOrigin);
WK_EXPORT WKStringRef WKSecurityOriginCopyHost(WKSecurityOriginRef securityOrigin);
WK_EXPORT unsigned short WKSecurityOriginGetPort(WKSecurityOriginRef securityOrigin);

#ifdef __cplusplus
}
#endif

#endif /* WKSecurityOriginRef_h */
