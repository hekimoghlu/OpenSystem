/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 2, 2024.
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

#include	<stdio.h>
#include	<stdlib.h>
#include	<string.h>
#include	<unixio.h>
#include	"vim.h"
int main(int argc, char *argv[])
{
    FILE	*fpi, *fpo;
    char	cmd[132], buf[BUFSIZ], *argp, *error_file, target[132], *mms;
    int		err = 0, err_line = 0;

    mms = "mms";
    argc--;
    argv++;
    while (argc-- > 0)
    {
	argp = *argv++;
	if (*argp == '-')
	{
	    switch (*++argp)
	    {
		case 'm':
		    mms = ++argp;
		    break;
		case 'e':
		    if (!*(error_file = ++argp))
		    {
			error_file = *argv++;
			argc--;
		    }
		    break;
		default:
		    break;
	    }
	}
	else
	{
	    if (*target)
		strcat(target, " ");
	    strcat(target, argp);
	}
    }
    vim_snprintf(cmd, sizeof(cmd), "%s/output=tmp:errors.vim_tmp %s",
								 mms, target);
    system(cmd);
    fpi = fopen("tmp:errors.vim_tmp", "r");
    fpo = fopen(error_file, "w");
    while (fgets(buf, BUFSIZ, fpi))
    {
	if (!memcmp(buf, "%CC-", 4))
	{
	    err_line++;
	    buf[strlen(buf)-1] = '\0';
	    err++;
	}
	else
	{
	    if (err_line)
	    {
		if (strstr(buf, _("At line")))
		{
		    err_line = 0;
		    fprintf(fpo, "@");
		}
		else
		    buf[strlen(buf)-1] = '\0';
	    }
	}
	fprintf(fpo, "%s", buf);
    }
    fclose(fpi);
    fclose(fpo);
    while (!delete("tmp:errors.vim_tmp"))
	/*nop*/;
    exit(err ? 44 : 1);
    return(0);
}
