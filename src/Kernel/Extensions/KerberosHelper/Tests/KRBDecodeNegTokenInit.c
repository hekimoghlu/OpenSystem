/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 20, 2022.
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
#include <KerberosHelper/KerberosHelper.h>
#include <CoreFoundation/CoreFoundation.h>
#include <GSS/gssapi.h>
#include <err.h>

int
main(int argc, char **argv)
{
    gss_buffer_desc empty = { 0, NULL }, out;
    OM_uint32 maj_stat, min_stat;
    gss_ctx_id_t ctx = GSS_C_NO_CONTEXT;

    maj_stat = gss_accept_sec_context(&min_stat, &ctx, GSS_C_NO_CREDENTIAL,
				      &empty, GSS_C_NO_CHANNEL_BINDINGS,
				      NULL, NULL, &out, NULL, NULL, NULL);
    if (maj_stat != GSS_S_CONTINUE_NEEDED)
	errx(1, "gss_accept_sec_context");

    CFDataRef data = CFDataCreateWithBytesNoCopy(NULL, out.value, out.length, kCFAllocatorNull);

    CFDictionaryRef dict = KRBDecodeNegTokenInit(NULL, data);
    if (dict == NULL)
	errx(1, "KRBDecodeNegTokenInit");

    CFShow(dict);
    CFRelease(dict);

    CFRelease(data);
    
    gss_release_buffer(&min_stat, &out);
    gss_delete_sec_context(&min_stat, &ctx, NULL);

    return 0;
}
