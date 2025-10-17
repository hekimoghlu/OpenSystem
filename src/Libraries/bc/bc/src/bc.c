/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 5, 2024.
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
#if BC_ENABLED

#include <string.h>

#include <bc.h>
#include <vm.h>

/**
 * The main function for bc.
 * @param argc  The number of arguments.
 * @param argv  The arguments.
 */
BcStatus
bc_main(int argc, char* argv[])
{
	// All of these just set bc-specific items in BcVm.

	vm->read_ret = BC_INST_RET;
	vm->help = bc_help;
	vm->sigmsg = bc_sig_msg;
	vm->siglen = bc_sig_msg_len;

	vm->next = bc_lex_token;
	vm->parse = bc_parse_parse;
	vm->expr = bc_parse_expr;

	return bc_vm_boot(argc, argv);
}
#endif // BC_ENABLED
