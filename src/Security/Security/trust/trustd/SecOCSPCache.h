/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 29, 2022.
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
/*!
    @header SecOCSPCache
    The functions provided in SecOCSPCache.h provide an interface to
    an OCSP caching module.
*/

#ifndef _SECURITY_SECOCSPCACHE_H_
#define _SECURITY_SECOCSPCACHE_H_

#include "trust/trustd/SecOCSPRequest.h"
#include "trust/trustd/SecOCSPResponse.h"
#include <CoreFoundation/CFURL.h>

__BEGIN_DECLS

void SecOCSPCacheReplaceResponse(SecOCSPResponseRef old_response,
    SecOCSPResponseRef response, CFURLRef localResponderURI, CFAbsoluteTime verifyTime);

SecOCSPResponseRef SecOCSPCacheCopyMatching(SecOCSPRequestRef request,
    CFURLRef localResponderURI /* may be NULL */);

SecOCSPResponseRef SecOCSPCacheCopyMatchingWithMinInsertTime(SecOCSPRequestRef request,
    CFURLRef localResponderURI, CFAbsoluteTime minInsertTime);

bool SecOCSPCacheFlush(CFErrorRef *error);

/* for testing purposes only */
bool SecOCSPCacheDeleteContent(CFErrorRef *error);
void SecOCSPCacheDeleteCache(void);
void SecOCSPCacheCloseDB(void);
CFStringRef SecOCSPCacheCopyPath(void);

__END_DECLS

#endif /* _SECURITY_SECOCSPCACHE_H_ */
