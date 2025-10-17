/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 27, 2021.
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
#include "curl_setup.h"

#ifdef CURL_MACOS_CALL_COPYPROXIES

#include <curl/curl.h>

#include "macos.h"

#include <SystemConfiguration/SCDynamicStoreCopySpecific.h>

CURLcode Curl_macos_init(void)
{
  {
    /*
     * The automagic conversion from IPv4 literals to IPv6 literals only
     * works if the SCDynamicStoreCopyProxies system function gets called
     * first. As Curl currently doesn't support system-wide HTTP proxies, we
     * therefore don't use any value this function might return.
     *
     * This function is only available on macOS and is not needed for
     * IPv4-only builds, hence the conditions for defining
     * CURL_MACOS_CALL_COPYPROXIES in curl_setup.h.
     */
    CFDictionaryRef dict = SCDynamicStoreCopyProxies(NULL);
    if(dict)
      CFRelease(dict);
  }
  return CURLE_OK;
}

#endif
