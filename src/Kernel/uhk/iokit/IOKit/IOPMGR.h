/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 2, 2022.
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
#pragma once

#include <machine/machine_routines.h>

#include <stdint.h>
#include <IOKit/IOService.h>

/*!
 * @class      IOPMGR
 * @abstract   The base class for power managers, such as ApplePMGR.
 */
class IOPMGR : public IOService
{
	OSDeclareAbstractStructors(IOPMGR);

public:
	/*!
	 * @function        enableCPUCore
	 * @abstract        Enable a single CPU core.
	 * @discussion      Release a secondary CPU core from reset, and enable
	 *                  external IRQ delivery to the core.  XNU will not
	 *                  invoke this method on the boot CPU's cpu_id.
	 * @param cpu_id    Logical CPU ID of the core.
	 * @param entry_pa  Physical address to use as the reset vector on the
	 *                  secondary CPU.  Not all platforms will honor this
	 *                  parameter; on Apple Silicon RVBAR_EL1 is programmed
	 *                  by iBoot.
	 */
	virtual void enableCPUCore(unsigned int cpu_id, uint64_t entry_pa);

	/*!
	 * @function      enableCPUCore
	 * @abstract      Deprecated - Enable a single CPU core.
	 */
	virtual void enableCPUCore(unsigned int cpu_id);

	/*!
	 * @function      disableCPUCore
	 * @abstract      Disable a single CPU core.
	 * @discussion    Prepare a secondary CPU core for power down, and
	 *                disable external IRQ delivery to the core.  XNU
	 *                will not invoke this method on the boot CPU's cpu_id.
	 *                Note that the enable and disable operations are not
	 *                symmetric, as disableCPUCore doesn't actually cut
	 *                power to the core.
	 * @param cpu_id  Logical CPU ID of the core.
	 */
	virtual void disableCPUCore(unsigned int cpu_id) = 0;

	/*!
	 * @function          enableCPUCluster
	 * @abstract          Enable power to a cluster of CPUs.
	 * @discussion        Called to power up a CPU cluster if the cluster-wide
	 *                    voltage rails are disabled (i.e. PIO to the cluster
	 *                    isn't even working).
	 * @param cluster_id  Cluster ID.
	 */
	virtual void enableCPUCluster(unsigned int cluster_id) = 0;

	/*!
	 * @function          disableCPUCluster
	 * @abstract          Disable power to a cluster of CPUs.
	 * @discussion        Called to disable the voltage rails on a CPU
	 *                    cluster.  This will only be invoked if all CPUs
	 *                    in the cluster are already disabled.  It is
	 *                    presumed that after this operation completes,
	 *                    PIO operations to the cluster will cause a
	 *                    fatal bus error.
	 * @param cluster_id  Cluster ID.
	 */
	virtual void disableCPUCluster(unsigned int cluster_id) = 0;

	/*!
	 * @function                   initCPUIdle
	 * @abstract                   Initialize idle-related parameters.
	 * @param info                 Pointer to the ml_processor_info_t struct that is
	 *                             being initialized (and hasn't been registered yet).
	 */
	virtual void initCPUIdle(ml_processor_info_t *info) = 0;

	/*!
	 * @function                   enterCPUIdle
	 * @abstract                   Called from cpu_idle() prior to entering the idle state on
	 *                             the current CPU.
	 * @param newIdleTimeoutTicks  If non-NULL, will be overwritten with a new idle timeout value,
	 *                             in ticks.  If the value is 0, XNU will disable the idle timer.
	 */
	virtual void enterCPUIdle(UInt64 *newIdleTimeoutTicks) = 0;

	/*!
	 * @function                   exitCPUIdle
	 * @abstract                   Called from cpu_idle_exit() after leaving the idle state on
	 *                             the current CPU.
	 * @param newIdleTimeoutTicks  If non-NULL, will be overwritten with a new idle timeout value,
	 *                             in ticks.  If the value is 0, XNU will disable the idle timer.
	 */
	virtual void exitCPUIdle(UInt64 *newIdleTimeoutTicks) = 0;

	/*!
	 * @function                   updateCPUIdle
	 * @abstract                   Called from timer_intr() to ask when to schedule the next idle
	 *                             timeout on the current CPU.
	 * @param newIdleTimeoutTicks  If non-NULL, will be overwritten with a new idle timeout value,
	 *                             in ticks.  If the value is 0, XNU will disable the idle timer.
	 */
	virtual void updateCPUIdle(UInt64 *newIdleTimeoutTicks) = 0;
};
