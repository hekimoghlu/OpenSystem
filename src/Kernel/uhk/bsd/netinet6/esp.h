/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 18, 2023.
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
/*	$FreeBSD: src/sys/netinet6/esp.h,v 1.2.2.2 2001/07/03 11:01:49 ume Exp $	*/
/*	$KAME: esp.h,v 1.16 2000/10/18 21:28:00 itojun Exp $	*/

/*
 * Copyright (C) 1995, 1996, 1997, and 1998 WIDE Project.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the project nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE PROJECT AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE PROJECT OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

/*
 * RFC1827/2406 Encapsulated Security Payload.
 */

#ifndef _NETINET6_ESP_H_
#define _NETINET6_ESP_H_
#include <sys/appleapiopts.h>
#include <sys/types.h>

struct esp {
	u_int32_t       esp_spi;        /* ESP */
	/*variable size, 32bit bound*/	/* Initialization Vector */
	/*variable size*/		/* Payload data */
	/*variable size*/		/* padding */
	/*8bit*/			/* pad size */
	/*8bit*/			/* next header */
	/*8bit*/			/* next header */
	/*variable size, 32bit bound*/	/* Authentication data (new IPsec) */
};

struct newesp {
	u_int32_t       esp_spi;        /* ESP */
	u_int32_t       esp_seq;        /* Sequence number */
	/*variable size*/		/* (IV and) Payload data */
	/*variable size*/		/* padding */
	/*8bit*/			/* pad size */
	/*8bit*/			/* next header */
	/*8bit*/			/* next header */
	/*variable size, 32bit bound*/	/* Authentication data */
};

struct esptail {
	u_int8_t        esp_padlen;     /* pad length */
	u_int8_t        esp_nxt;        /* Next header */
	/*variable size, 32bit bound*/	/* Authentication data (new IPsec)*/
};

#ifdef BSD_KERNEL_PRIVATE
struct secasvar;

#define ESP_AUTH_MAXSUMSIZE   64

#define ESP_ASSERT(_cond, _format, ...)                                                  \
	do {                                                                             \
	        if (__improbable(!(_cond))) {                                            \
	                panic("%s:%d " _format, __FUNCTION__, __LINE__, ##__VA_ARGS__);  \
	        }                                                                        \
	} while (0)

#define ESP_CHECK_ARG(_arg) ESP_ASSERT(_arg != NULL, #_arg " is NULL")

#define _esp_log(_level, _format, ...)  \
	log(_level, "%s:%d " _format, __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define esp_log_err(_format, ...) _esp_log(LOG_ERR, _format, ##__VA_ARGS__)
#define esp_log_default(_format, ...) _esp_log(LOG_NOTICE, _format, ##__VA_ARGS__)
#define esp_log_info(_format, ...) _esp_log(LOG_INFO, _format, ##__VA_ARGS__)

#define _esp_packet_log(_level, _format, ...)  \
	ipseclog((_level, "%s:%d " _format, __FUNCTION__, __LINE__, ##__VA_ARGS__))
#define esp_packet_log_err(_format, ...) _esp_packet_log(LOG_ERR, _format, ##__VA_ARGS__)

struct esp_algorithm {
	uint32_t padbound;        /* pad boundary, in byte */
	int ivlenval;           /* iv length, in byte */
	int (*mature)(struct secasvar *);
	u_int16_t keymin;     /* in bits */
	u_int16_t keymax;     /* in bits */
	size_t (*schedlen)(const struct esp_algorithm *);
	const char *name;
	int (*ivlen)(const struct esp_algorithm *, struct secasvar *);
	int (*decrypt)(struct mbuf *, size_t,
	    struct secasvar *, const struct esp_algorithm *, int);
	int (*encrypt)(struct mbuf *, size_t, size_t,
	    struct secasvar *, const struct esp_algorithm *, int);
	/* not supposed to be called directly */
	int (*schedule)(const struct esp_algorithm *, struct secasvar *);
	int (*blockdecrypt)(const struct esp_algorithm *,
	    struct secasvar *, u_int8_t *, u_int8_t *);
	int (*blockencrypt)(const struct esp_algorithm *,
	    struct secasvar *, u_int8_t *, u_int8_t *);
	/* For Authenticated Encryption Methods */
	size_t icvlen;
	int (*finalizedecrypt)(struct secasvar *, u_int8_t *, size_t);
	int (*finalizeencrypt)(struct secasvar *, u_int8_t *, size_t);
	int (*encrypt_pkt)(struct secasvar *, uint8_t *, size_t,
	    struct newesp *, uint8_t *, size_t, uint8_t *, size_t);
	int (*decrypt_pkt)(struct secasvar *, uint8_t *, size_t,
	    struct newesp *, uint8_t *, size_t, uint8_t *, size_t);
};

extern os_log_t esp_mpkl_log_object;

extern const struct esp_algorithm *esp_algorithm_lookup(int);
extern int esp_max_ivlen(void);

/* crypt routines */
extern int esp4_output(struct mbuf *, struct secasvar *);
extern void esp4_input(struct mbuf *, int off);
extern struct mbuf *esp4_input_extended(struct mbuf *, int off, ifnet_t interface);
extern size_t esp_hdrsiz(struct ipsecrequest *);
extern int esp_kpipe_output(struct secasvar *, kern_packet_t, kern_packet_t);
extern int esp_kpipe_input(ifnet_t, kern_packet_t, kern_packet_t);

extern int esp_schedule(const struct esp_algorithm *, struct secasvar *);
extern int esp_auth(struct mbuf *, size_t, size_t,
    struct secasvar *, u_char *__sized_by(ESP_AUTH_MAXSUMSIZE));
extern int esp_auth_data(struct secasvar *, uint8_t *, size_t, uint8_t *, size_t);

extern void esp_init(void);
#endif /* BSD_KERNEL_PRIVATE */

#endif /* _NETINET6_ESP_H_ */
