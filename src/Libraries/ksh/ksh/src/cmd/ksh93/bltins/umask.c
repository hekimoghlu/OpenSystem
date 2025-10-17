/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 31, 2021.
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
#pragma prototyped
/*
 * umask [-S] [mask]
 *
 *   David Korn
 *   AT&T Labs
 *   research!dgk
 *
 */

#include	<ast.h>	
#include	<sfio.h>	
#include	<error.h>	
#include	<ctype.h>	
#include	<ls.h>	
#include	<shell.h>	
#include	"builtins.h"
#ifndef SH_DICT
#   define SH_DICT	"libshell"
#endif

int	b_umask(int argc,char *argv[],Shbltin_t *context)
{
	register char *mask;
	register int flag = 0, sflag = 0;
	NOT_USED(context);
	while((argc = optget(argv,sh_optumask))) switch(argc)
	{
		case 'S':
			sflag++;
			break;
		case ':':
			errormsg(SH_DICT,2, "%s", opt_info.arg);
			break;
		case '?':
			errormsg(SH_DICT,ERROR_usage(2), "%s",opt_info.arg);
			break;
	}
	if(error_info.errors)
		errormsg(SH_DICT,ERROR_usage(2),"%s",optusage((char*)0));
	argv += opt_info.index;
	if(mask = *argv)
	{
		register int c;	
		if(isdigit(*mask))
		{
			while(c = *mask++)
			{
				if (c>='0' && c<='7')	
					flag = (flag<<3) + (c-'0');	
				else
					errormsg(SH_DICT,ERROR_exit(1),e_number,*argv);
			}
		}
		else
		{
			char *cp = mask;
			flag = umask(0);
			c = strperm(cp,&cp,~flag&0777);
			if(*cp)
			{
				umask(flag);
				errormsg(SH_DICT,ERROR_exit(1),e_format,mask);
			}
			flag = (~c&0777);
		}
		umask(flag);	
	}	
	else
	{
		umask(flag=umask(0));
		if(sflag)
			sfprintf(sfstdout,"%s\n",fmtperm(~flag&0777));
		else
			sfprintf(sfstdout,"%0#4o\n",flag);
	}
	return(0);
}

