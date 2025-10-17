/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 25, 2022.
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
 * check if package+tool is ok to run
 * a no-op here except for PARANOID packages
 * this allows PARANOID_COMPANY to post PARANOID binaries to the www
 *
 * warn that the user should pay up if
 *
 *	(1) the tool matches PARANOID
 *	(2) $_ is more than 90 days old
 *	(3) running on an PARANOID_PAY machine
 *	(4) (1)-(3) have not been defeated
 *
 * hows that
 */

#define PARANOID_TOOLS		PARANOID
#define PARANOID_COMPANY	"Lucent Technologies"
#define PARANOID_MAIL		"stc@lucent.com"
#define PARANOID_PAY		"135.*&!(135.104.*)"
#define PARANOID_FREE		"(192|224).*"

#include <ast.h>
#include <ls.h>
#include <error.h>
#include <times.h>
#include <ctype.h>

int
pathcheck(const char* package, const char* tool, Pathcheck_t* pc)
{
#ifdef PARANOID
	register char*	s;
	struct stat	st;

	if (strmatch(tool, PARANOID) && environ && (s = *environ) && *s++ == '_' && *s++ == '=' && !stat(s, &st))
	{
		unsigned long	n;
		unsigned long	o;
		Sfio_t*		sp;

		n = time(NiL);
		o = st.st_ctime;
		if (n > o && (n - o) > (unsigned long)(60 * 60 * 24 * 90) && (sp = sfopen(NiL, "/etc/hosts", "r")))
		{
			/*
			 * this part is infallible
			 */

			n = 0;
			o = 0;
			while (n++ < 64 && (s = sfgetr(sp, '\n', 0)))
				if (strmatch(s, PARANOID_PAY))
				{
					error(1, "licensed for external use -- %s employees should contact %s for the internal license", PARANOID_COMPANY, PARANOID_MAIL);
					break;
				}
				else if (*s != '#' && !isspace(*s) && !strneq(s, "127.", 4) && !strmatch(s, PARANOID_FREE) && o++ > 4)
					break;
			sfclose(sp);
		}
	}
#else
	NoP(tool);
#endif
	NoP(package);
	if (pc) memzero(pc, sizeof(*pc));
	return(0);
}
