/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 6, 2025.
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
 * Copyright 2008 Sun Microsystems, Inc.  All rights reserved.
 * Use is subject to license terms.
 */
/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 5, 2025.
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
 *//*	  All Rights Reserved  	*/

#include <libelf.h>
#include "decl.h"

/*
 * Global data
 * _elf_encode		Host/internal data encoding.  If the host has
 *			an encoding that matches one known for the
 *			ELF format, this changes.  An machine with an
 *			unknown encoding keeps ELFDATANONE and forces
 *			conversion for host/target translation.
 * _elf_work		Working version given to the lib by application.
 *			See elf_version().
 * _elf_globals_mutex	mutex to protect access to all global data items.
 */

/*
 * __libc_threaded is a private symbol exported from libc in Solaris 10.
 * It is used to tell if we are running in a threaded world or not.
 * Between Solaris 2.5 and Solaris 9, this was named __threaded.
 * The name had to be changed because the Sun Workshop 6 update 1
 * compilation system used it to mean "we are linked with libthread"
 * rather than its true meaning in Solaris 10, "more than one thread exists".
 */
// XXX_PRAGMA_WEAK #pragma weak		__libc_threaded
extern int		__libc_threaded;

unsigned		_elf_encode = ELFDATANONE;
const Snode32		_elf32_snode_init;
const Snode64		_elf64_snode_init;
unsigned		_elf_work = EV_NONE;
mutex_t			_elf_globals_mutex = DEFAULTMUTEX;

int			*_elf_libc_threaded = &__libc_threaded;

