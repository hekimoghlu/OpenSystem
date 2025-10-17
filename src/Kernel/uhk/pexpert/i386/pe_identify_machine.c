/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 29, 2022.
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
#include <machine/machine_routines.h>
#include <pexpert/pexpert.h>
#include <pexpert/protos.h>

/* Local declarations */
void pe_identify_machine(boot_args *args);

/* pe_identify_machine:
 *
 *   Sets up platform parameters.
 *   Returns:    nothing
 */
void
pe_identify_machine(__unused boot_args *args)
{
	// Clear the gPEClockFrequencyInfo struct
	bzero((void *)&gPEClockFrequencyInfo, sizeof(clock_frequency_info_t));

	// Start with default values.
	gPEClockFrequencyInfo.timebase_frequency_hz = 1000000000;
	gPEClockFrequencyInfo.bus_frequency_hz      =  100000000;
	gPEClockFrequencyInfo.cpu_frequency_hz      =  300000000;

	gPEClockFrequencyInfo.bus_frequency_min_hz = gPEClockFrequencyInfo.bus_frequency_hz;
	gPEClockFrequencyInfo.bus_frequency_max_hz = gPEClockFrequencyInfo.bus_frequency_hz;
	gPEClockFrequencyInfo.cpu_frequency_min_hz = gPEClockFrequencyInfo.cpu_frequency_hz;
	gPEClockFrequencyInfo.cpu_frequency_max_hz = gPEClockFrequencyInfo.cpu_frequency_hz;

	gPEClockFrequencyInfo.dec_clock_rate_hz = gPEClockFrequencyInfo.timebase_frequency_hz;
	gPEClockFrequencyInfo.bus_clock_rate_hz = gPEClockFrequencyInfo.bus_frequency_hz;
	gPEClockFrequencyInfo.cpu_clock_rate_hz = gPEClockFrequencyInfo.cpu_frequency_hz;

	// Get real number from some where.

	// Set the num / den pairs form the hz values.
	gPEClockFrequencyInfo.bus_clock_rate_num = gPEClockFrequencyInfo.bus_clock_rate_hz;
	gPEClockFrequencyInfo.bus_clock_rate_den = 1;

	gPEClockFrequencyInfo.bus_to_cpu_rate_num =
	    (2 * gPEClockFrequencyInfo.cpu_clock_rate_hz) / gPEClockFrequencyInfo.bus_clock_rate_hz;
	gPEClockFrequencyInfo.bus_to_cpu_rate_den = 2;

	gPEClockFrequencyInfo.bus_to_dec_rate_num = 1;
	gPEClockFrequencyInfo.bus_to_dec_rate_den =
	    gPEClockFrequencyInfo.bus_clock_rate_hz / gPEClockFrequencyInfo.dec_clock_rate_hz;
}

void
PE_panic_hook(const char *str __unused)
{
}
