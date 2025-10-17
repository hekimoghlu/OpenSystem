/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 5, 2025.
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
 * Glenn Fowler
 * at&t Research
 *
 * coshell export var set/unset
 */

#include "colib.h"

/*
 * set or unset coshell export variable
 */

int
coexport(Coshell_t* co, const char* name, const char* value)
{
	Coexport_t*	ex;
	char*		v;

	if (!co->export)
	{
		if (!(co->exdisc = vmnewof(co->vm, 0, Dtdisc_t, 1, 0)))
			return -1;
		co->exdisc->link = offsetof(Coexport_t, link);
		co->exdisc->key = offsetof(Coexport_t, name);
		co->exdisc->size = 0;
		if (!(co->export = dtnew(co->vm, co->exdisc, Dtset)))
		{
			vmfree(co->vm, co->exdisc);
			return -1;
		}
	}
	if (!(ex = (Coexport_t*)dtmatch(co->export, name)))
	{
		if (!value)
			return 0;
		if (!(ex = vmnewof(co->vm, 0, Coexport_t, 1, strlen(name))))
			return -1;
		strcpy(ex->name, name);
		dtinsert(co->export, ex);
	}
	if (ex->value)
	{
		vmfree(co->vm, ex->value);
		ex->value = 0;
	}
	if (value)
	{
		if (!(v = vmstrdup(co->vm, value)))
			return -1;
		ex->value = v;
	}
	else
	{
		dtdelete(co->export, ex);
		vmfree(co->vm, ex);
	}
	co->init.sync = 1;
	return 0;
}
