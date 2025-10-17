/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 27, 2023.
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
 * standalone mini error implementation
 */

#include <ast.h>
#include <error.h>

Error_info_t	error_info;

void
errorv(const char* id, int level, va_list ap)
{
	char*	a;
	char*	s;
	int	flags;

	if (level < 0)
		flags = 0;
	else
	{
		flags = level & ~ERROR_LEVEL;
		level &= ERROR_LEVEL;
	}
	a = va_arg(ap, char*);
	if (level && ((s = error_info.id) || (s = (char*)id)))
	{
		if (!(flags & ERROR_USAGE))
			sfprintf(sfstderr, "%s: ", s);
		else if (strcmp(a, "%s"))
			sfprintf(sfstderr, "Usage: %s ", s);
	}
	if (flags & ERROR_USAGE)
		/*nop*/;
	else if (level < 0)
		sfprintf(sfstderr, "debug%d: ", level);
	else if (level)
	{
		if (level == ERROR_WARNING)
		{
			sfprintf(sfstderr, "warning: ");
			error_info.warnings++;
		}
		else
		{
			error_info.errors++;
			if (level == ERROR_PANIC)
				sfprintf(sfstderr, "panic: ");
		}
		if (error_info.line)
		{
			if (error_info.file && *error_info.file)
				sfprintf(sfstderr, "\"%s\", ", error_info.file);
			sfprintf(sfstderr, "line %d: ", error_info.line);
		}
	}
	sfvprintf(sfstderr, a, ap);
	sfprintf(sfstderr, "\n");
	if (level >= ERROR_FATAL)
		exit(level - ERROR_FATAL + 1);
}

void
error(int level, ...)
{
	va_list	ap;

	va_start(ap, level);
	errorv(NiL, level, ap);
	va_end(ap);
}

int
errorf(void* handle, void* discipline, int level, ...)
{
	va_list	ap;

	va_start(ap, level);
	errorv((discipline && handle) ? *((char**)handle) : (char*)handle, level, ap);
	va_end(ap);
	return 0;
}
