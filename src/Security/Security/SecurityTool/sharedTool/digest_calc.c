/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 15, 2024.
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
#include "builtin_commands.h"

#include <stdlib.h>
#include <strings.h>
#include <unistd.h>

#include "SecurityTool/sharedTool/readline.h"

#include <corecrypto/ccsha1.h>
#include <corecrypto/ccsha2.h>

#include <AssertMacros.h>

extern int command_digest(int argc, char * const *argv)
{
    int result = 1, fd;
    const struct ccdigest_info *di;
    unsigned char *digest = NULL;
    unsigned long i,j;
    size_t nr = 0, totalBytes = 0;
    char data [getpagesize()];
    
    if (argc < 3)
        return SHOW_USAGE_MESSAGE;
    
    if (strcasecmp("sha1", argv[1]) == 0)
    {
        //printf("Calculating sha1\n");
        di = ccsha1_di();
    }
    else if (strcasecmp("sha256", argv[1]) == 0)
    {
        //printf("Calculating sha256\n");
        di = ccsha256_di();
    }
    else if (strcasecmp("sha512", argv[1]) == 0)
    {
        //printf("Calculating sha256\n");
        di = ccsha512_di();
        
    }
    else
        return SHOW_USAGE_MESSAGE;
    
    ccdigest_di_decl(di, ctx);
    
    digest = malloc(di->output_size);
    require_quiet(digest, exit);
    
    for (i = 2; i < (unsigned int)argc; ++i)
    {
        printf("%s(%s)= ", argv[1], argv[i]);
        if ((fd = inspect_file_and_size(argv[i], NULL)) == -1)
        {
            printf("error reading file\n");
            continue;
        }

        ccdigest_init(di, ctx);

        totalBytes = 0;
        while((nr = pread(fd, data, sizeof(data), totalBytes)) > 0){
            ccdigest_update(di, ctx, nr, data);
            totalBytes += nr;
        }
    
        ccdigest_final(di, ctx, digest);

        for (j = 0; j < di->output_size; j++)
            printf("%02x", digest[j]);
        printf("\n");
    }
    result = 0;
    
exit:
    if (digest)
        free(digest);
    
    return result;
}
