/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 30, 2022.
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
#include <atm/atm_internal.h>
#include <machine/commpage.h>
#include <pexpert/pexpert.h>

#if CONFIG_EXCLAVES
extern kern_return_t exclaves_oslog_set_trace_mode(uint32_t);
#endif // CONFIG_EXCLAVES

/*
 * Global that is set by diagnosticd and readable by userspace
 * via the commpage.
 */
static uint32_t atm_diagnostic_config;
static bool disable_atm;

/*
 * Routine: atm_init
 * Purpose: Initialize the atm subsystem.
 * Returns: None.
 */
void
atm_init(void)
{
	char temp_buf[20];

	/* Disable atm if disable_atm present in device-tree properties or in boot-args */
	if ((PE_get_default("kern.disable_atm", temp_buf, sizeof(temp_buf))) ||
	    (PE_parse_boot_argn("-disable_atm", temp_buf, sizeof(temp_buf)))) {
		disable_atm = true;
	}

	if (!PE_parse_boot_argn("atm_diagnostic_config", &atm_diagnostic_config, sizeof(atm_diagnostic_config))) {
		if (!PE_get_default("kern.atm_diagnostic_config", &atm_diagnostic_config, sizeof(atm_diagnostic_config))) {
			atm_diagnostic_config = 0;
		}
	}

	kprintf("ATM subsystem is initialized\n");
}

/*
 * Routine: atm_reset
 * Purpose: re-initialize the atm subsystem (e.g. for userspace reboot)
 * Returns: None.
 */
void
atm_reset(void)
{
	atm_init();
	commpage_update_atm_diagnostic_config(atm_diagnostic_config);
#if CONFIG_EXCLAVES
	exclaves_oslog_set_trace_mode(atm_diagnostic_config);
#endif // CONFIG_EXCLAVES
}

/*
 * Routine: atm_set_diagnostic_config
 * Purpose: Set global atm_diagnostic_config and update the commpage to reflect
 *          the new value.
 * Returns: Error if ATM is disabled.
 */
kern_return_t
atm_set_diagnostic_config(uint32_t diagnostic_config)
{
	if (disable_atm) {
		return KERN_NOT_SUPPORTED;
	}

	atm_diagnostic_config = diagnostic_config;
	commpage_update_atm_diagnostic_config(atm_diagnostic_config);
#if CONFIG_EXCLAVES
	return exclaves_oslog_set_trace_mode(diagnostic_config);
#else
	return KERN_SUCCESS;
#endif // CONFIG_EXCLAVES
}

/*
 * Routine: atm_get_diagnostic_config
 * Purpose: Get global atm_diagnostic_config.
 * Returns: Diagnostic value
 */
uint32_t
atm_get_diagnostic_config(void)
{
	return atm_diagnostic_config;
}
