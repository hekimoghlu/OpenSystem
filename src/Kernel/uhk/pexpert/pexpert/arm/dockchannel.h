/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 14, 2023.
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
#ifndef _PEXPERT_ARM_DOCKCHANNEL_H
#define _PEXPERT_ARM_DOCKCHANNEL_H

#define DOCKCHANNEL_UART                        (1)
#define DOCKCHANNEL_STRIDE                      (0x10000)

// Channel index
#define DOCKCHANNEL_UART_CHANNEL                (0)

/* Dock Agent Interrupt Control Register */
#define rDOCKCHANNELS_AGENT_AP_INTR_CTRL(_agent_base)        ((uintptr_t) ((_agent_base) + 0x00))

/* When this bit is set, the write watermark interrupt of Dock Channel 0 on the device side is enabled */
#define DC0_WR_DEVICE_EN        (1U << 0)
/* When this bit is set, the read watermark interrupt of Dock Channel 0 on the device side is enabled */
#define DC0_RD_DEVICE_EN        (1U << 1)

/* Dock Agent Interrupt Status Register */
#define rDOCKCHANNELS_AGENT_AP_INTR_STATUS(_agent_base)      ((uintptr_t) ((_agent_base) + 0x04))

/**
 * This bit is set when the write watermark interrupt of Dock Channel 0 on the device side is
 * asserted. This bit remains set until cleared by SW by writing a 1.
 */
#define DC0_WR_DEVICE_STAT      (1U << 0)
/**
 * This bit is set when the read watermark interrupt of Dock Channel 0 on the device side is
 * asserted. This bit remains set until cleared by SW by writing a 1.
 */
#define DC0_RD_DEVICE_STAT      (1U << 1)

/* Dock Agent Error Interrupt Control Register */
#define rDOCKCHANNELS_AGENT_AP_ERR_INTR_CTRL(_agent_base)    ((uintptr_t) ((_agent_base) + 0x08))

/* When this bit is set, the error interrupt of Dock Channel 0 on the device side is enabled */
#define DC0_ERROR_DEVICE_EN     (1U << 0)
/* When this bit is set, the error interrupt of Dock Channel 0 on the dock side is enabled */
#define DC0_ERROR_DOCK_EN       (1U << 1)

/* Dock Agent Error Interrupt Status Register */
#define rDOCKCHANNELS_AGENT_AP_ERR_INTR_STATUS(_agent_base)  ((uintptr_t) ((_agent_base) + 0x0c))
/**
 * This bit is set when the error interrupt of Dock Channel 0 on the device side is asserted.
 * This bit remains set until cleared by SW by writing a 1.
 */
#define DC0_ERR_DEVICE_STAT     (1U << 0)
/**
 * This bit is set when the error interrupt of Dock Channel 0 on the dock side is asserted.
 * This bit remains set until cleared by SW by writing a 1.
 */
#define DC0_ERR_DOCK_STAT       (1U << 1)

#define rDOCKCHANNELS_DEV_WR_WATERMARK(_base, _ch)     ((uintptr_t) ((_base) + ((_ch) * DOCKCHANNEL_STRIDE) + 0x0000))
#define rDOCKCHANNELS_DEV_RD_WATERMARK(_base, _ch)     ((uintptr_t) ((_base) + ((_ch) * DOCKCHANNEL_STRIDE) + 0x0004))
#define rDOCKCHANNELS_DEV_DRAIN_CFG(_base, _ch)        ((uintptr_t) ((_base) + ((_ch) * DOCKCHANNEL_STRIDE) + 0x0008))

#define rDOCKCHANNELS_DEV_WDATA1(_base, _ch)           ((uintptr_t) ((_base) + ((_ch) * DOCKCHANNEL_STRIDE) + 0x4004))
#define rDOCKCHANNELS_DEV_WSTAT(_base, _ch)            ((uintptr_t) ((_base) + ((_ch) * DOCKCHANNEL_STRIDE) + 0x4014))
#define rDOCKCHANNELS_DEV_RDATA0(_base, _ch)           ((uintptr_t) ((_base) + ((_ch) * DOCKCHANNEL_STRIDE) + 0x4018))
#define rDOCKCHANNELS_DEV_RDATA1(_base, _ch)           ((uintptr_t) ((_base) + ((_ch) * DOCKCHANNEL_STRIDE) + 0x401c))

#define rDOCKCHANNELS_DOCK_RDATA1(_base, _ch)          ((uintptr_t) ((_base) + ((_ch) * DOCKCHANNEL_STRIDE) + 0xc01c))
#define rDOCKCHANNELS_DOCK_RDATA3(_base, _ch)          ((uintptr_t) ((_base) + ((_ch) * DOCKCHANNEL_STRIDE) + 0xc024))

#endif  /* !_PEXPERT_ARM_DOCKCHANNEL_H */
