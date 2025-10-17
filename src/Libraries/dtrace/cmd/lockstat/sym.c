/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 30, 2024.
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
#include <CoreSymbolication/CoreSymbolication.h>
#include <CoreSymbolication/CoreSymbolicationPrivate.h>

static CSSymbolicatorRef 	g_symbolicator;

int
symtab_init(void)
{
	uint32_t symflags = 0x0;
	symflags |= kCSSymbolicatorDefaultCreateFlags;
	symflags |= kCSSymbolicatorUseSlidKernelAddresses;

	/* retrieve the kernel symbolicator */
	g_symbolicator = CSSymbolicatorCreateWithMachKernelFlagsAndNotification(symflags, NULL);
	if (CSIsNull(g_symbolicator)) {
		fprintf(stderr, "could not retrieve the kernel symbolicator\n");
		return -1;
	}

	return 0;
}

char const*
addr_to_sym(uintptr_t addr, uintptr_t *offset, size_t *sizep)
{
	CSSymbolRef symbol;
	CSRange	range;

	assert(offset);
	assert(sizep);

	symbol = CSSymbolicatorGetSymbolWithAddressAtTime(g_symbolicator, addr, kCSNow);
	if (!CSIsNull(symbol)) {
		range = CSSymbolGetRange(symbol);
		*offset = addr - range.location;
		*sizep = range.length;
		return CSSymbolGetName(symbol);
	}

	return NULL;
}

uintptr_t
sym_to_addr(char *name)
{
	CSSymbolRef symbol;

	symbol = CSSymbolicatorGetSymbolWithNameAtTime(g_symbolicator, name, kCSNow);
	if (!CSIsNull(symbol))
		return CSSymbolGetRange(symbol).location;

	return NULL;
}

size_t
sym_size(char *name)
{
	CSSymbolRef symbol;

	symbol = CSSymbolicatorGetSymbolWithNameAtTime(g_symbolicator, name, kCSNow);
	if (!CSIsNull(symbol))
		return CSSymbolGetRange(symbol).length;

	return NULL;
}

