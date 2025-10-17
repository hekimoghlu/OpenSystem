/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 16, 2025.
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
#ifndef _PEXPERT_ARM_PROTOS_H
#define _PEXPERT_ARM_PROTOS_H

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

extern vm_offset_t pe_arm_get_soc_base_phys(void);
extern uint32_t pe_arm_init_interrupts(void *args);
extern void pe_arm_debug_init_early(void *args);
extern void pe_arm_debug_init_late(void);

#ifdef  PEXPERT_KERNEL_PRIVATE
extern void console_write_unbuffered(char);
#endif
int serial_init(void);

/**
 * Forbid or allow transmission over each serial until they receive data.
 */
void
serial_set_on_demand(bool);

#if HIBERNATION
void serial_hibernation_init(void);
#endif /* HIBERNATION */
int serial_getc(void);
void serial_putc(char);
void uart_putc(char);
#ifdef PRIVATE
void serial_putc_options(char, bool);
void uart_putc_options(char, bool);
#endif /* PRIVATE */
int uart_getc(void);

void pe_init_fiq(void);

#ifdef PRIVATE
/**
 * One hot ids to distinquish between all supported serial devices
 */
typedef enum serial_device {
	SERIAL_UNKNOWN=0x0,
	SERIAL_APPLE_UART=0x1,
	SERIAL_DOCKCHANNEL=0x2,
	SERIAL_PL011_UART=0x4,
	SERIAL_DCC_UART=0x8
} serial_device_t;

kern_return_t serial_irq_enable(serial_device_t device);
kern_return_t serial_irq_action(serial_device_t device);
bool serial_irq_filter(serial_device_t device);

void serial_go_to_sleep(void);
#endif /* PRIVATE */

int switch_to_serial_console(void);
void switch_to_old_console(int);

#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif /* _PEXPERT_ARM_PROTOS_H */
