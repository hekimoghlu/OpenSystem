/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 14, 2024.
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
#ifndef lf_hfs_raw_read_write_h
#define lf_hfs_raw_read_write_h

#include "lf_hfs_vnode.h"
#include "lf_hfs.h"

errno_t  raw_readwrite_read_mount( vnode_t psMountVnode, uint64_t uBlockN, uint64_t uClusterSize, void* pvBuf, uint64_t uBufLen, uint64_t *piActuallyRead, uint64_t* puReadStartCluster );
errno_t  raw_readwrite_write_mount( vnode_t psMountVnode, uint64_t uBlockN, uint64_t uClusterSize, void* pvBuf, uint64_t uBufLen, uint64_t *piActuallyWritten, uint64_t* puWrittenStartCluster );

int      raw_readwrite_get_cluster_from_offset( vnode_t psVnode, uint64_t uWantedOffset, uint64_t* puStartCluster, uint64_t* puInClusterOffset, uint64_t* puContigousClustersInBytes );
errno_t  raw_readwrite_write( vnode_t psVnode, uint64_t uOffset, void* pvBuf, uint64_t uLength, uint64_t *piActuallyWritten );
errno_t  raw_readwrite_write_internal( vnode_t psVnode, uint64_t uCluster, uint64_t uContigousClustersInBytes,
                                     uint64_t Offset, uint64_t uBytesToWrite, void* pvBuf, uint64_t *piActuallyWritten );
errno_t  raw_readwrite_read( vnode_t psVnode, uint64_t uOffset, void* pvBuf, uint64_t uLength, size_t *piActuallyRead, uint64_t* puReadStartCluster );
errno_t  raw_readwrite_read_internal( vnode_t psVnode, uint64_t uStartCluster, uint64_t uContigousClustersInBytes,
                                      uint64_t Offset, uint64_t uBytesToRead, void* pvBuf, uint64_t *piActuallyRead );

int         raw_readwrite_zero_fill_init( void );
void        raw_readwrite_zero_fill_de_init( void );
int         raw_readwrite_zero_fill_fill( hfsmount_t* psMount, uint64_t uOffset, uint32_t uLength );
errno_t     raw_readwrite_zero_fill_last_block_suffix( vnode_t psVnode );


#endif /* lf_hfs_raw_read_write_h */
