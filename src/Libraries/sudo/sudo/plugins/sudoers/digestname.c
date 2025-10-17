/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 10, 2021.
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
/*
 * This is an open source non-commercial project. Dear PVS-Studio, please check it.
 * PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
 */

#include <config.h>

#include "sudo_compat.h"
#include "sudo_digest.h"
#include "sudoers_debug.h"
#include "parse.h"

const char *
digest_type_to_name(int digest_type)
{
    const char *digest_name;
    debug_decl(digest_type_to_name, SUDOERS_DEBUG_UTIL);

    switch (digest_type) {
    case SUDO_DIGEST_SHA224:
	digest_name = "sha224";
	break;
    case SUDO_DIGEST_SHA256:
	digest_name = "sha256";
	break;
    case SUDO_DIGEST_SHA384:
	digest_name = "sha384";
	break;
    case SUDO_DIGEST_SHA512:
	digest_name = "sha512";
	break;
    default:
	digest_name = "unknown digest";
	break;
    }
    debug_return_const_str(digest_name);
}
