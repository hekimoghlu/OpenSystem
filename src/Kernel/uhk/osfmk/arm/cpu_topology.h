/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 30, 2023.
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
/**
 * Data structures that reveal the CPU topology of the system. This header is
 * used to contain the API which is kernel private for now, since things are
 * constantly evolving and kexts that depend on it could break unexpectedly.
 * It should not be included directly but instead is included from
 * machine_routines.h
 */

#ifndef _ARM_CPU_TOPOLOGY_H_
#define _ARM_CPU_TOPOLOGY_H_

#include <stdint.h>
__BEGIN_DECLS

/*!
 * @typedef ml_topology_cpu_t
 * @brief Describes one CPU core in the topology.
 *
 * @field cpu_id            Logical CPU ID: 0, 1, 2, 3, 4, ...
 *                          Dynamically assigned by XNU so it might not match EDT.  No holes.
 * @field phys_id           Physical CPU ID (EDT: reg).  Same as MPIDR[15:0], i.e.
 *                          (cluster_id << 8) | core_number_within_cluster
 * @field cluster_id        Logical Cluster ID: 0, 1, 2, 3, 4, ...
 *                          Dynamically assigned by XNU so it might not match EDT.  No holes.
 * @field die_id            Die ID (EDT: die-id)
 * @field cluster_type      The type of CPUs found in this cluster.
 * @field l2_cache_size     Size of the L2 cache, in bytes.  0 if unknown or not present.
 * @field l2_cache_id       l2-cache-id property read from EDT.
 * @field l3_cache_size     Size of the L3 cache, in bytes.  0 if unknown or not present.
 * @field l3_cache_id       l3-cache-id property read from EDT.
 * @field cpu_IMPL_regs     IO-mapped virtual address of cpuX_IMPL (implementation-defined) register block.
 * @field cpu_IMPL_pa       Physical address of cpuX_IMPL register block.
 * @field cpu_IMPL_len      Length of cpuX_IMPL register block.
 * @field cpu_UTTDBG_regs   IO-mapped virtual address of cpuX_UTTDBG register block.
 * @field cpu_UTTDBG_pa     Physical address of cpuX_UTTDBG register block, if set in DT, else zero
 * @field cpu_UTTDBG_len    Length of cpuX_UTTDBG register block, if set in DT, else zero
 * @field coresight_regs    IO-mapped virtual address of CoreSight debug register block.
 * @field coresight_pa      Physical address of CoreSight register block.
 * @field coresight_len     Length of CoreSight register block.
 * @field die_cluster_id    Physical cluster ID within the local die (EDT: die-cluster-id)
 * @field cluster_core_id   Physical core ID within the local cluster (EDT: cluster-core-id)
 */
typedef struct ml_topology_cpu {
	unsigned int                    cpu_id;
	uint32_t                        phys_id;
	unsigned int                    cluster_id;
	unsigned int                    die_id;
	cluster_type_t                  cluster_type;
	uint32_t                        l2_access_penalty; /* unused */
	uint32_t                        l2_cache_size;
	uint32_t                        l2_cache_id;
	uint32_t                        l3_cache_size;
	uint32_t                        l3_cache_id;
	vm_offset_t                     cpu_IMPL_regs;
	uint64_t                        cpu_IMPL_pa;
	uint64_t                        cpu_IMPL_len;
	vm_offset_t                     cpu_UTTDBG_regs;
	uint64_t                        cpu_UTTDBG_pa;
	uint64_t                        cpu_UTTDBG_len;
	vm_offset_t                     coresight_regs;
	uint64_t                        coresight_pa;
	uint64_t                        coresight_len;
	unsigned int                    die_cluster_id;
	unsigned int                    cluster_core_id;
} ml_topology_cpu_t;

/*!
 * @typedef ml_topology_cluster_t
 * @brief Describes one cluster in the topology.
 *
 * @field cluster_id        Cluster ID (EDT: cluster-id)
 * @field cluster_type      The type of CPUs found in this cluster.
 * @field num_cpus          Total number of usable CPU cores in this cluster.
 * @field first_cpu_id      The cpu_id of the first CPU in the cluster.
 * @field cpu_mask          A bitmask representing the cpu_id's that belong to the cluster.  Example:
 *                          If the cluster contains CPU4 and CPU5, cpu_mask will be 0x30.
 * @field die_id            Die ID.
 * @field die_cluster_id    Physical cluster ID within the local die (EDT: die-cluster-id)
 * @field acc_IMPL_regs     IO-mapped virtual address of acc_IMPL (implementation-defined) register block.
 * @field acc_IMPL_pa       Physical address of acc_IMPL register block.
 * @field acc_IMPL_len      Length of acc_IMPL register block.
 * @field cpm_IMPL_regs     IO-mapped virtual address of cpm_IMPL (implementation-defined) register block.
 * @field cpm_IMPL_pa       Physical address of cpm_IMPL register block.
 * @field cpm_IMPL_len      Length of cpm_IMPL register block.
 */
typedef struct ml_topology_cluster {
	unsigned int                    cluster_id;
	cluster_type_t                  cluster_type;
	unsigned int                    num_cpus;
	unsigned int                    first_cpu_id;
	uint64_t                        cpu_mask;
	unsigned int                    die_id;
	unsigned int                    die_cluster_id;
	vm_offset_t                     acc_IMPL_regs;
	uint64_t                        acc_IMPL_pa;
	uint64_t                        acc_IMPL_len;
	vm_offset_t                     cpm_IMPL_regs;
	uint64_t                        cpm_IMPL_pa;
	uint64_t                        cpm_IMPL_len;
} ml_topology_cluster_t;

// Bump this version number any time any ml_topology_* struct changes in a
// way that is likely to break kexts, so that users can check whether their
// headers are compatible with the running kernel
#define CPU_TOPOLOGY_VERSION 1

/*!
 * @typedef ml_topology_info_t
 * @brief Describes the CPU topology for all APs in the system.  Populated from EDT and read-only at runtime.
 * @discussion This struct only lists CPU cores that are considered usable by both iBoot and XNU.  Some
 *             physically present CPU cores may be considered unusable due to configuration options like
 *             the "cpus=" boot-arg.  Cores that are disabled in hardware will not show up in EDT at all, so
 *             they also will not be present in this struct.
 *
 * @field version            Version of the struct (set to CPU_TOPOLOGY_VERSION).
 * @field num_cpus           Total number of usable CPU cores.
 * @field max_cpu_id         The highest usable logical CPU ID.
 * @field num_clusters       Total number of AP CPU clusters on the system (usable or not).
 * @field max_cluster_id     The highest cluster ID found in EDT.
 * @field cpus               List of |num_cpus| entries.
 * @field clusters           List of |num_clusters| entries.
 * @field boot_cpu           Points to the |cpus| entry for the boot CPU.
 * @field boot_cluster       Points to the |clusters| entry which contains the boot CPU.
 * @field chip_revision      Silicon revision reported by iBoot, which comes from the
 *                           SoC-specific fuse bits.  See CPU_VERSION_xx macros for definitions.
 * @field cluster_power_down Set to 1 if there exists at least one cluster on the system that can be
 *                           power-gated at runtime.
 */
typedef struct ml_topology_info {
	unsigned int                    version;
	unsigned int                    num_cpus;
	unsigned int                    max_cpu_id;
	unsigned int                    num_clusters;
	unsigned int                    max_cluster_id;
	unsigned int                    max_die_id;
	ml_topology_cpu_t               *cpus;
	ml_topology_cluster_t           *clusters;
	ml_topology_cpu_t               *boot_cpu;
	ml_topology_cluster_t           *boot_cluster;
	unsigned int                    chip_revision;
	unsigned int                    cluster_types;
	unsigned int                    cluster_type_num_cpus[MAX_CPU_TYPES];
	unsigned int                    cluster_type_num_clusters[MAX_CPU_TYPES];
	unsigned int                    cluster_power_down;
} ml_topology_info_t;

/*!
 * @function ml_get_topology_info
 * @result A pointer to the read-only topology struct.  Does not need to be freed.  Returns NULL
 *         if the struct hasn't been initialized or the feature is unsupported.
 */
const ml_topology_info_t *ml_get_topology_info(void);

__END_DECLS

#endif /* _ARM_CPU_TOPOLOGY_H_ */
