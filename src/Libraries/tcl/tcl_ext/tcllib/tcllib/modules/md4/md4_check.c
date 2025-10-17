/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 19, 2025.
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
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <openssl/md4.h>

static const char rcsid[] = 
"$Id: md4_check.c,v 1.2 2004/01/15 06:36:13 andreas_kupries Exp $";

void
md4(const char *buf, size_t len, unsigned char *res)
{
    MD4_CTX ctx;
    MD4_Init(&ctx);
    MD4_Update(&ctx, buf, len);
    MD4_Final(res, &ctx);
}

void
dump(unsigned char *data, size_t len)
{
    char buf[80], *p;
    size_t cn, n;

    for (cn = 0, p = buf; cn < len; cn++, p += 2) {
        n = sprintf(p, "%02X", data[cn]);
    }
    puts(buf);
}

int
main(int argc, char *argv[])
{
    size_t cn;
    char buf[256];
    unsigned char r[16];

    memset(buf, 'a', 256);

    for (cn = 0; cn < 150; cn++) {
        md4(buf, cn, r);
        printf("%7d ", cn);
        dump(r, 16);
    }
    return 0;
}

/*
 * Local variables:
 *   mode: c
 *   indent-tabs-mode: nil
 * End:
 */
