/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 4, 2021.
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
#if DC_ENABLED

#include <string.h>

#include <dc.h>
#include <vm.h>

/**
 * The main function for dc.
 * @param argc  The number of arguments.
 * @param argv  The arguments.
 */
BcStatus
dc_main(int argc, char* argv[])
{
	// All of these just set dc-specific items in BcVm.

	vm->read_ret = BC_INST_POP_EXEC;
	vm->help = dc_help;
	vm->sigmsg = dc_sig_msg;
	vm->siglen = dc_sig_msg_len;

	vm->next = dc_lex_token;
	vm->parse = dc_parse_parse;
	vm->expr = dc_parse_expr;

	return bc_vm_boot(argc, argv);
}
#endif // DC_ENABLED
