/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 8, 2022.
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
#ifndef SUDO_LBUF_H
#define SUDO_LBUF_H

/*
 * Line buffer struct.
 */
struct sudo_lbuf {
    int (*output)(const char *);
    char *buf;
    const char *continuation;
    unsigned int indent;
    unsigned int len;
    unsigned int size;
    unsigned short cols;
    unsigned short error;
};

typedef int (*sudo_lbuf_output_t)(const char *);

/* Flags for sudo_lbuf_append_esc() */
#define LBUF_ESC_CNTRL	0x01
#define LBUF_ESC_BLANK	0x02
#define LBUF_ESC_QUOTE	0x04

sudo_dso_public void sudo_lbuf_init_v1(struct sudo_lbuf *lbuf, sudo_lbuf_output_t output, int indent, const char *continuation, int cols);
sudo_dso_public void sudo_lbuf_destroy_v1(struct sudo_lbuf *lbuf);
sudo_dso_public bool sudo_lbuf_append_v1(struct sudo_lbuf *lbuf, const char *fmt, ...) sudo_printflike(2, 3);
sudo_dso_public bool sudo_lbuf_append_esc_v1(struct sudo_lbuf *lbuf, int flags, const char *fmt, ...) sudo_printflike(3, 4);
sudo_dso_public bool sudo_lbuf_append_quoted_v1(struct sudo_lbuf *lbuf, const char *set, const char *fmt, ...) sudo_printflike(3, 4);
sudo_dso_public void sudo_lbuf_print_v1(struct sudo_lbuf *lbuf);
sudo_dso_public bool sudo_lbuf_error_v1(struct sudo_lbuf *lbuf);
sudo_dso_public void sudo_lbuf_clearerr_v1(struct sudo_lbuf *lbuf);

#define sudo_lbuf_init(_a, _b, _c, _d, _e) sudo_lbuf_init_v1((_a), (_b), (_c), (_d), (_e))
#define sudo_lbuf_destroy(_a) sudo_lbuf_destroy_v1((_a))
#define sudo_lbuf_append sudo_lbuf_append_v1
#define sudo_lbuf_append_esc sudo_lbuf_append_esc_v1
#define sudo_lbuf_append_quoted sudo_lbuf_append_quoted_v1
#define sudo_lbuf_print(_a) sudo_lbuf_print_v1((_a))
#define sudo_lbuf_error(_a) sudo_lbuf_error_v1((_a))
#define sudo_lbuf_clearerr(_a) sudo_lbuf_clearerr_v1((_a))

#endif /* SUDO_LBUF_H */
