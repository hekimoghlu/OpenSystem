/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 24, 2023.
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
#include "config.h"
#include "WebKitURIUtilities.h"

#include <wtf/URLHelpers.h>
#include <wtf/text/CString.h>
#include <wtf/text/WTFString.h>

/**
 * webkit_uri_for_display:
 * @uri: the URI to be converted
 *
 * Use this function to format a URI for display.
 *
 * The URIs used internally by
 * WebKit may contain percent-encoded characters or Punycode, which are not
 * generally suitable to display to users. This function provides protection
 * against IDN homograph attacks, so in some cases the host part of the returned
 * URI may be in Punycode if the safety check fails.
 *
 * Returns: (nullable) (transfer full): @uri suitable for display, or %NULL in
 *    case of error.
 *
 * Since: 2.24
 */
gchar* webkit_uri_for_display(const gchar* uri)
{
    g_return_val_if_fail(uri, nullptr);

    String result = WTF::URLHelpers::userVisibleURL(uri);
    if (!result)
        return nullptr;

    return g_strdup(result.utf8().data());
}
