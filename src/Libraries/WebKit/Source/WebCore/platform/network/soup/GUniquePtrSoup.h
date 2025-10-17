/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 11, 2023.
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
#pragma once

#include <wtf/Platform.h>

#if USE(SOUP)

#include <libsoup/soup.h>
#include <wtf/glib/GUniquePtr.h>

namespace WTF {

WTF_DEFINE_GPTR_DELETER(SoupCookie, soup_cookie_free)
#if SOUP_CHECK_VERSION(2, 67, 1)
WTF_DEFINE_GPTR_DELETER(SoupHSTSPolicy, soup_hsts_policy_free)
#endif
#if USE(SOUP2)
WTF_DEFINE_GPTR_DELETER(SoupURI, soup_uri_free)
#endif
#if SOUP_CHECK_VERSION(2, 99, 3)
WTF_DEFINE_GPTR_DELETER(SoupMessageHeaders, soup_message_headers_unref)
#else
WTF_DEFINE_GPTR_DELETER(SoupMessageHeaders, soup_message_headers_free)
#endif

} // namespace WTF

#endif // USE(SOUP)
