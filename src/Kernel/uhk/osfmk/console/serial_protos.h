/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 27, 2024.
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
 * @OSF_COPYRIGHT@
 */
/*
 * @APPLE_FREE_COPYRIGHT@
 */

#ifndef _CONSOLE_SERIAL_PROTOS_H_
#define _CONSOLE_SERIAL_PROTOS_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

void serial_keyboard_init(void);
void serial_keyboard_start(void) __dead2;
void serial_keyboard_poll(void) __dead2;

/* Boot serial mode (see bits below). */
extern uint32_t serialmode;

/* Is output supported ? */
#define SERIALMODE_OUTPUT    0x01

/* Is input supported ? */
#define SERIALMODE_INPUT     0x02

/* Force synchronous output ? */
#define SERIALMODE_SYNCDRAIN 0x04

/* Load Base/Recovery/FVUnlock TTY */
#define SERIALMODE_BASE_TTY  0x08

/* Prevent IOLogs writing to serial */
#define SERIALMODE_NO_IOLOG  0x10

/* Allow DriverKit os_log/IOLogs writing to serial */
#define SERIALMODE_DKLOG     0x20

/** Start logging on to a serial only once data has been received on this
 * serial. Requires SERIALMODE_INPUT set, ingored otherwise.  */
#define SERIALMODE_ON_DEMAND  0x40

#if CONFIG_EXCLAVES
/* Prevent Exclave Logs writing to serial */
#define SERIALMODE_NO_EXCLAVE  0x80
#endif

extern uint32_t cons_ops_index;
extern const uint32_t nconsops;

/* disable_serial_output disables kprintf() *and* unbuffered panic output. */
extern bool disable_serial_output;

/* Shortcuts for serialmode & {SERIALMODE_NOIOLOG, SERIALMODE_DKLOG}. */
extern bool disable_iolog_serial_output;
extern bool enable_dklog_serial_output;

void console_init(void);

int _serial_getc(bool wait);
int _vcgetc(bool wait);

struct console_ops {
	void (*putc)(char, bool);
	int (*getc)(bool);
};

boolean_t console_is_serial(void);
int switch_to_serial_console(void);
int switch_to_video_console(void);
void switch_to_old_console(int old_console);

#define SERIAL_CONS_OPS 0
#define VC_CONS_OPS 1

#ifdef XNU_KERNEL_PRIVATE

#define SERIAL_CONS_BUF_SIZE  256
struct console_printbuf_state {
	int pos;
	int total;
	int flags;
#define CONS_PB_WRITE_NEWLINE  0x1
#define CONS_PB_CANBLOCK       0x2
	char str[SERIAL_CONS_BUF_SIZE];
};

extern int console_printbuf_drain_initialized;
void console_printbuf_state_init(struct console_printbuf_state * data, int write_on_newline, int can_block);
void console_printbuf_putc(int ch, void *arg);
void console_printbuf_clear(struct console_printbuf_state * info);
int console_write_try(char * str, int size);


#endif /* XNU_KERNEL_PRIVATE */

#ifdef __cplusplus
}
#endif


#endif /* _CONSOLE_SERIAL_PROTOS_H_ */
