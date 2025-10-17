/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 27, 2024.
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
#include <stdbool.h>
#if DEVELOPMENT || DEBUG

#include <kern/startup.h>
#include <kern/zalloc.h>
#include <sys/proc_ro.h>
#include <sys/vm.h>

#include <tests/ktest.h>

#include <libkern/tree.h>

#include <sys/socket.h>
#include <net/if_dl.h>
#include <net/ndrv.h>
#include <netinet/if_ether.h>
#include <sys/kern_control.h>
#include <sys/un.h>
#include <net/sockaddr_utils.h>

struct sockaddr *sockaddr_ptr;
const struct sockaddr *sockaddr_const_ptr;

struct sockaddr src_sockaddr = {
	.sa_len = 16,
	.sa_family = AF_UNSPEC,
	.sa_data = {0, }
};
struct sockaddr *sockaddr_ptr = &src_sockaddr;
const struct sockaddr *sockaddr_const_ptr = &src_sockaddr;


struct sockaddr_storage src_sockaddr_storage = {
	.ss_len = 255,
	.ss_family = AF_UNSPEC
};
struct sockaddr_storage *sockaddr_storage_ptr = &src_sockaddr_storage;
const struct sockaddr_storage *sockaddr_storage_const_ptr = &src_sockaddr_storage;


struct sockaddr_ctl src_sockaddr_ctl = {
	.sc_len = 16,
	.sc_family = AF_SYSTEM
};
struct sockaddr_ctl *sockaddr_ctl_ptr = &src_sockaddr_ctl;
const struct sockaddr_ctl *sockaddr_ctl_const_ptr = &src_sockaddr_ctl;


struct sockaddr_dl src_sockaddr_dl = {
	.sdl_len = 16,
	.sdl_family = AF_LINK,
	.sdl_index = 0,
};
struct sockaddr_dl *sockaddr_dl_ptr = &src_sockaddr_dl;
const struct sockaddr_dl *sockaddr_dl_const_ptr = &src_sockaddr_dl;



struct sockaddr_in src_sockaddr_in = {
	.sin_len = 16,
	.sin_family = AF_INET,
	.sin_addr = {.s_addr = 0001001001001 },
	.sin_port = 1111,
	.sin_zero = {0, }
};
struct sockaddr_in *sockaddr_in_ptr = &src_sockaddr_in;
const struct sockaddr_in *sockaddr_in_const_ptr = &src_sockaddr_in;


struct sockaddr_in6 src_sockaddr_in6 = {
	.sin6_len = 28,
	.sin6_family = AF_INET6,
	.sin6_flowinfo = 100,
};
struct sockaddr_in6 *sockaddr_in6_ptr = &src_sockaddr_in6;
const struct sockaddr_in6 *sockaddr_in6_const_ptr = &src_sockaddr_in6;



struct sockaddr_inarp src_sockaddr_inarp = {
	.sin_len = 16,
	.sin_family = AF_INET,
	.sin_addr = {.s_addr = 0001001001001 },
	.sin_srcaddr = {.s_addr = 0001001001001 },
	.sin_tos = 1,
	.sin_other = 2
};

struct sockaddr_inarp *sockaddr_inarp_ptr = &src_sockaddr_inarp;
const struct sockaddr_inarp *sockaddr_inarp_const_ptr = &src_sockaddr_inarp;


struct sockaddr_inifscope src_sockaddr_inifscope = {
	.sin_len = 16,
	.sin_family = AF_INET,
	.sin_addr = {.s_addr = 0001001001001 },
	.sin_port = 1111,
};
struct sockaddr_inifscope *sockaddr_inifscope_ptr = &src_sockaddr_inifscope;
const struct sockaddr_inifscope *sockaddr_inifscope_const_ptr = &src_sockaddr_inifscope;


struct sockaddr_ndrv src_sockaddr_ndrv = {
	.snd_len = 16,
	.snd_family = AF_NDRV,
};
struct sockaddr_ndrv *sockaddr_ndrv_ptr = &src_sockaddr_ndrv;
const struct sockaddr_ndrv *sockaddr_ndrv_const_ptr = &src_sockaddr_ndrv;


struct sockaddr_sys src_sockaddr_sys = {
	.ss_len = 16,
	.ss_family = AF_SYSTEM
};
struct sockaddr_sys *sockaddr_sys_ptr = &src_sockaddr_sys;
const struct sockaddr_sys *sockaddr_sys_const_ptr = &src_sockaddr_sys;



struct sockaddr_un src_sockaddr_un = {
	.sun_len = 106,
	.sun_family = AF_UNIX
};
struct sockaddr_un *sockaddr_un_ptr = &src_sockaddr_un;
const struct sockaddr_un *sockaddr_un_const_ptr = &src_sockaddr_un;


union sockaddr_in_4_6 src_sockaddr_in_4_6 = {
	.sa = {
		.sa_len = 28,
		.sa_family = AF_INET6
	},
};
union sockaddr_in_4_6 *sockaddr_in_4_6_ptr = &src_sockaddr_in_4_6;
const union sockaddr_in_4_6 *sockaddr_in_4_6_const_ptr = &src_sockaddr_in_4_6;


#define BYTE_INPUT_SIZE 256

char src_bytes[BYTE_INPUT_SIZE];
char * bytes_ptr __sized_by(BYTE_INPUT_SIZE) = &src_bytes[0];
const char * bytes_const_ptr __sized_by(BYTE_INPUT_SIZE) = &src_bytes[0];

uint8_t src_uint8_t[BYTE_INPUT_SIZE];
uint8_t * uint8_t_ptr __sized_by(BYTE_INPUT_SIZE) = &src_uint8_t[0];
const uint8_t * uint8_t_const_ptr __sized_by(BYTE_INPUT_SIZE) = &src_uint8_t[0];


__attribute__((__noinline__))
static void
validate_conversion_result(const void *result)
{
	T_ASSERT_NOTNULL(result, "The conversion is invalid at %s:%u", __FILE__, __LINE__);
}


static void
test_valid_conversions_to_bytes()
{
	uint8_t *result;
	const uint8_t *const_result;

	/* Non-const to non-const conversions. */
	result = __SA_UTILS_CONV_TO_BYTES(sockaddr_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_BYTES(sockaddr_storage_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_BYTES(sockaddr_ctl_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_BYTES(sockaddr_dl_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_BYTES(sockaddr_in_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_BYTES(sockaddr_in6_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_BYTES(sockaddr_inarp_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_BYTES(sockaddr_inifscope_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_BYTES(sockaddr_ndrv_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_BYTES(sockaddr_sys_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_BYTES(sockaddr_un_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_BYTES(sockaddr_in_4_6_ptr);
	validate_conversion_result(result);

	/* Const to const conversions. */
	const_result = __SA_UTILS_CONV_TO_BYTES(sockaddr_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_BYTES(sockaddr_storage_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_BYTES(sockaddr_ctl_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_BYTES(sockaddr_dl_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_BYTES(sockaddr_in_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_BYTES(sockaddr_in6_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_BYTES(sockaddr_inarp_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_BYTES(sockaddr_inifscope_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_BYTES(sockaddr_ndrv_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_BYTES(sockaddr_sys_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_BYTES(sockaddr_un_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_BYTES(sockaddr_in_4_6_const_ptr);
	validate_conversion_result(const_result);
}

static void
test_valid_conversions_to_sockaddr()
{
	struct sockaddr *result;
	const struct sockaddr *const_result;

	/* Non-const to non-const conversions. */
	result = __SA_UTILS_CONV_TO_SOCKADDR(sockaddr_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_SOCKADDR(sockaddr_storage_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_SOCKADDR(sockaddr_ctl_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_SOCKADDR(sockaddr_dl_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_SOCKADDR(sockaddr_in_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_SOCKADDR(sockaddr_in6_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_SOCKADDR(sockaddr_inarp_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_SOCKADDR(sockaddr_inifscope_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_SOCKADDR(sockaddr_ndrv_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_SOCKADDR(sockaddr_sys_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_SOCKADDR(sockaddr_un_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_SOCKADDR(bytes_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_SOCKADDR(uint8_t_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_SOCKADDR(sockaddr_in_4_6_ptr);
	validate_conversion_result(result);


	/* Const to const conversions. */
	const_result = __SA_UTILS_CONV_TO_SOCKADDR(sockaddr_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_SOCKADDR(sockaddr_storage_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_SOCKADDR(sockaddr_ctl_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_SOCKADDR(sockaddr_dl_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_SOCKADDR(sockaddr_in_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_SOCKADDR(sockaddr_in6_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_SOCKADDR(sockaddr_inarp_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_SOCKADDR(sockaddr_inifscope_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_SOCKADDR(sockaddr_ndrv_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_SOCKADDR(sockaddr_sys_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_SOCKADDR(sockaddr_un_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_SOCKADDR(bytes_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_SOCKADDR(uint8_t_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_SOCKADDR(sockaddr_in_4_6_const_ptr);
	validate_conversion_result(const_result);

	/* Const to non-const (a.k.a. "deconst") conversions. */
	result = __SA_UTILS_DECONST_AND_CONV_TO_SOCKADDR(sockaddr_const_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_DECONST_AND_CONV_TO_SOCKADDR(sockaddr_storage_const_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_DECONST_AND_CONV_TO_SOCKADDR(sockaddr_ctl_const_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_DECONST_AND_CONV_TO_SOCKADDR(sockaddr_dl_const_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_DECONST_AND_CONV_TO_SOCKADDR(sockaddr_in_const_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_DECONST_AND_CONV_TO_SOCKADDR(sockaddr_in6_const_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_DECONST_AND_CONV_TO_SOCKADDR(sockaddr_inarp_const_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_DECONST_AND_CONV_TO_SOCKADDR(sockaddr_inifscope_const_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_DECONST_AND_CONV_TO_SOCKADDR(sockaddr_ndrv_const_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_DECONST_AND_CONV_TO_SOCKADDR(sockaddr_sys_const_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_DECONST_AND_CONV_TO_SOCKADDR(sockaddr_un_const_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_DECONST_AND_CONV_TO_SOCKADDR(sockaddr_in_4_6_const_ptr);
	validate_conversion_result(result);
}

static void
test_valid_conversions_to_sockaddr_ctl()
{
	struct sockaddr_ctl *result;
	const struct sockaddr_ctl *const_result;

	/* Non-const to non-const conversions. */
	result = __SA_UTILS_CONV_TO_SOCKADDR_CTL(sockaddr_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_SOCKADDR_CTL(sockaddr_storage_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_SOCKADDR_CTL(sockaddr_ctl_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_SOCKADDR_CTL(bytes_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_SOCKADDR_CTL(uint8_t_ptr);
	validate_conversion_result(result);

	/* Const to const conversions. */
	const_result = __SA_UTILS_CONV_TO_SOCKADDR_CTL(sockaddr_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_SOCKADDR_CTL(sockaddr_storage_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_SOCKADDR_CTL(sockaddr_ctl_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_SOCKADDR_CTL(bytes_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_SOCKADDR_CTL(uint8_t_const_ptr);
	validate_conversion_result(const_result);

	/* Const to non-const (a.k.a. "deconst") conversions. */
	result = __SA_UTILS_DECONST_AND_CONV_TO_SOCKADDR_CTL(sockaddr_const_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_DECONST_AND_CONV_TO_SOCKADDR_CTL(sockaddr_storage_const_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_DECONST_AND_CONV_TO_SOCKADDR_CTL(sockaddr_ctl_const_ptr);
	validate_conversion_result(result);
}

static void
test_valid_conversions_to_sockaddr_dl()
{
	struct sockaddr_dl *result;
	const struct sockaddr_dl *const_result;

	/* Non-const to non-const conversions. */
	result = __SA_UTILS_CONV_TO_SOCKADDR_DL(sockaddr_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_SOCKADDR_DL(sockaddr_storage_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_SOCKADDR_DL(sockaddr_dl_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_SOCKADDR_DL(bytes_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_SOCKADDR_DL(uint8_t_ptr);
	validate_conversion_result(result);

	/* Const to const conversions. */
	const_result = __SA_UTILS_CONV_TO_SOCKADDR_DL(sockaddr_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_SOCKADDR_DL(sockaddr_storage_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_SOCKADDR_DL(sockaddr_dl_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_SOCKADDR_DL(bytes_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_SOCKADDR_DL(uint8_t_const_ptr);
	validate_conversion_result(const_result);

	/* Const to non-const (a.k.a. "deconst") conversions. */
	result = __SA_UTILS_DECONST_AND_CONV_TO_SOCKADDR_DL(sockaddr_const_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_DECONST_AND_CONV_TO_SOCKADDR_DL(sockaddr_storage_const_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_DECONST_AND_CONV_TO_SOCKADDR_DL(sockaddr_dl_const_ptr);
	validate_conversion_result(result);
}

static void
test_valid_conversions_to_sockaddr_in()
{
	struct sockaddr_in *result;
	const struct sockaddr_in *const_result;

	/* Non-const to non-const conversions. */
	result = __SA_UTILS_CONV_TO_SOCKADDR_IN(sockaddr_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_SOCKADDR_IN(sockaddr_storage_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_SOCKADDR_IN(sockaddr_in_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_SOCKADDR_IN(sockaddr_in_4_6_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_SOCKADDR_IN(bytes_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_SOCKADDR_IN(uint8_t_ptr);
	validate_conversion_result(result);

	/* Const to const conversions. */
	const_result = __SA_UTILS_CONV_TO_SOCKADDR_IN(sockaddr_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_SOCKADDR_IN(sockaddr_storage_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_SOCKADDR_IN(sockaddr_in_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_SOCKADDR_IN(sockaddr_in_4_6_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_SOCKADDR_IN(bytes_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_SOCKADDR_IN(uint8_t_const_ptr);
	validate_conversion_result(const_result);

	/* Const to non-const (a.k.a. "deconst") conversions. */
	result = __SA_UTILS_DECONST_AND_CONV_TO_SOCKADDR_IN(sockaddr_const_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_DECONST_AND_CONV_TO_SOCKADDR_IN(sockaddr_storage_const_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_DECONST_AND_CONV_TO_SOCKADDR_IN(sockaddr_in_const_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_DECONST_AND_CONV_TO_SOCKADDR_IN(sockaddr_in_4_6_const_ptr);
	validate_conversion_result(result);
}

static void
test_valid_conversions_to_sockaddr_in6()
{
	struct sockaddr_in6 *result;
	const struct sockaddr_in6 *const_result;

	/* Non-const to non-const conversions. */
	result = __SA_UTILS_CONV_TO_SOCKADDR_IN6(sockaddr_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_SOCKADDR_IN6(sockaddr_storage_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_SOCKADDR_IN6(sockaddr_in6_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_SOCKADDR_IN6(sockaddr_in_4_6_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_SOCKADDR_IN6(bytes_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_SOCKADDR_IN6(uint8_t_ptr);
	validate_conversion_result(result);

	/* Const to const conversions. */
	const_result = __SA_UTILS_CONV_TO_SOCKADDR_IN6(sockaddr_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_SOCKADDR_IN6(sockaddr_storage_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_SOCKADDR_IN6(sockaddr_in6_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_SOCKADDR_IN6(sockaddr_in_4_6_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_SOCKADDR_IN6(bytes_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_SOCKADDR_IN6(uint8_t_const_ptr);
	validate_conversion_result(const_result);

	/* Const to non-const (a.k.a. "deconst") conversions. */
	result = __SA_UTILS_DECONST_AND_CONV_TO_SOCKADDR_IN6(sockaddr_const_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_DECONST_AND_CONV_TO_SOCKADDR_IN6(sockaddr_storage_const_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_DECONST_AND_CONV_TO_SOCKADDR_IN6(sockaddr_in6_const_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_DECONST_AND_CONV_TO_SOCKADDR_IN6(sockaddr_in_4_6_const_ptr);
	validate_conversion_result(result);
}

static void
test_valid_conversions_to_sockaddr_inarp()
{
	struct sockaddr_inarp *result;
	const struct sockaddr_inarp *const_result;

	/* Non-const to non-const conversions. */
	result = __SA_UTILS_CONV_TO_SOCKADDR_INARP(sockaddr_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_SOCKADDR_INARP(sockaddr_storage_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_SOCKADDR_INARP(sockaddr_inarp_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_SOCKADDR_INARP(bytes_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_SOCKADDR_INARP(uint8_t_ptr);
	validate_conversion_result(result);

	/* Const to const conversions. */
	const_result = __SA_UTILS_CONV_TO_SOCKADDR_INARP(sockaddr_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_SOCKADDR_INARP(sockaddr_storage_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_SOCKADDR_INARP(sockaddr_inarp_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_SOCKADDR_INARP(bytes_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_SOCKADDR_INARP(uint8_t_const_ptr);
	validate_conversion_result(const_result);

	/* Const to non-const (a.k.a. "deconst") conversions. */
	result = __SA_UTILS_DECONST_AND_CONV_TO_SOCKADDR_INARP(sockaddr_const_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_DECONST_AND_CONV_TO_SOCKADDR_INARP(sockaddr_storage_const_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_DECONST_AND_CONV_TO_SOCKADDR_INARP(sockaddr_inarp_const_ptr);
	validate_conversion_result(result);
}

static void
test_valid_conversions_to_sockaddr_inifscope()
{
	struct sockaddr_inifscope *result;
	const struct sockaddr_inifscope *const_result;

	/* Non-const to non-const conversions. */
	result = __SA_UTILS_CONV_TO_SOCKADDR_INIFSCOPE(sockaddr_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_SOCKADDR_INIFSCOPE(sockaddr_storage_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_SOCKADDR_INIFSCOPE(sockaddr_inifscope_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_SOCKADDR_INIFSCOPE(sockaddr_in_4_6_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_SOCKADDR_INIFSCOPE(bytes_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_SOCKADDR_INIFSCOPE(uint8_t_ptr);
	validate_conversion_result(result);

	/* Const to const conversions. */
	const_result = __SA_UTILS_CONV_TO_SOCKADDR_INIFSCOPE(sockaddr_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_SOCKADDR_INIFSCOPE(sockaddr_storage_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_SOCKADDR_INIFSCOPE(sockaddr_inifscope_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_SOCKADDR_INIFSCOPE(sockaddr_in_4_6_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_SOCKADDR_INIFSCOPE(bytes_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_SOCKADDR_INIFSCOPE(uint8_t_const_ptr);
	validate_conversion_result(const_result);

	/* Const to non-const (a.k.a. "deconst") conversions. */
	result = __SA_UTILS_DECONST_AND_CONV_TO_SOCKADDR_INIFSCOPE(sockaddr_const_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_DECONST_AND_CONV_TO_SOCKADDR_INIFSCOPE(sockaddr_storage_const_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_DECONST_AND_CONV_TO_SOCKADDR_INIFSCOPE(sockaddr_inifscope_const_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_DECONST_AND_CONV_TO_SOCKADDR_INIFSCOPE(sockaddr_in_4_6_const_ptr);
	validate_conversion_result(result);
}

static void
test_valid_conversions_to_sockaddr_ndrv()
{
	struct sockaddr_ndrv *result;
	const struct sockaddr_ndrv *const_result;

	/* Non-const to non-const conversions. */
	result = __SA_UTILS_CONV_TO_SOCKADDR_NDRV(sockaddr_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_SOCKADDR_NDRV(sockaddr_storage_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_SOCKADDR_NDRV(sockaddr_ndrv_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_SOCKADDR_NDRV(bytes_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_SOCKADDR_NDRV(uint8_t_ptr);
	validate_conversion_result(result);

	/* Const to const conversions. */
	const_result = __SA_UTILS_CONV_TO_SOCKADDR_NDRV(sockaddr_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_SOCKADDR_NDRV(sockaddr_storage_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_SOCKADDR_NDRV(sockaddr_ndrv_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_SOCKADDR_NDRV(bytes_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_SOCKADDR_NDRV(uint8_t_const_ptr);
	validate_conversion_result(const_result);

	/* Const to non-const (a.k.a. "deconst") conversions. */
	result = __SA_UTILS_DECONST_AND_CONV_TO_SOCKADDR_NDRV(sockaddr_const_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_DECONST_AND_CONV_TO_SOCKADDR_NDRV(sockaddr_storage_const_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_DECONST_AND_CONV_TO_SOCKADDR_NDRV(sockaddr_ndrv_const_ptr);
	validate_conversion_result(result);
}

static void
test_valid_conversions_to_sockaddr_sys()
{
	struct sockaddr_sys *result;
	const struct sockaddr_sys *const_result;

	/* Non-const to non-const conversions. */
	result = __SA_UTILS_CONV_TO_SOCKADDR_SYS(sockaddr_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_SOCKADDR_SYS(sockaddr_storage_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_SOCKADDR_SYS(sockaddr_sys_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_SOCKADDR_SYS(bytes_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_SOCKADDR_SYS(uint8_t_ptr);
	validate_conversion_result(result);

	/* Const to const conversions. */
	const_result = __SA_UTILS_CONV_TO_SOCKADDR_SYS(sockaddr_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_SOCKADDR_SYS(sockaddr_storage_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_SOCKADDR_SYS(sockaddr_sys_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_SOCKADDR_SYS(bytes_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_SOCKADDR_SYS(uint8_t_const_ptr);
	validate_conversion_result(const_result);

	/* Const to non-const (a.k.a. "deconst") conversions. */
	result = __SA_UTILS_DECONST_AND_CONV_TO_SOCKADDR_SYS(sockaddr_const_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_DECONST_AND_CONV_TO_SOCKADDR_SYS(sockaddr_storage_const_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_DECONST_AND_CONV_TO_SOCKADDR_SYS(sockaddr_sys_const_ptr);
	validate_conversion_result(result);
}

static void
test_valid_conversions_to_sockaddr_un()
{
	struct sockaddr_un *result;
	const struct sockaddr_un *const_result;

	/* Non-const to non-const conversions. */
	result = __SA_UTILS_CONV_TO_SOCKADDR_UN(sockaddr_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_SOCKADDR_UN(sockaddr_storage_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_SOCKADDR_UN(sockaddr_un_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_SOCKADDR_UN(bytes_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_CONV_TO_SOCKADDR_UN(uint8_t_ptr);
	validate_conversion_result(result);

	/* Const to const conversions. */
	const_result = __SA_UTILS_CONV_TO_SOCKADDR_UN(sockaddr_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_SOCKADDR_UN(sockaddr_storage_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_SOCKADDR_UN(sockaddr_un_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_SOCKADDR_UN(bytes_const_ptr);
	validate_conversion_result(const_result);

	const_result = __SA_UTILS_CONV_TO_SOCKADDR_UN(uint8_t_const_ptr);
	validate_conversion_result(const_result);

	/* Const to non-const (a.k.a. "deconst") conversions. */
	result = __SA_UTILS_DECONST_AND_CONV_TO_SOCKADDR_UN(sockaddr_const_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_DECONST_AND_CONV_TO_SOCKADDR_UN(sockaddr_storage_const_ptr);
	validate_conversion_result(result);

	result = __SA_UTILS_DECONST_AND_CONV_TO_SOCKADDR_UN(sockaddr_un_const_ptr);
	validate_conversion_result(result);
}

static int
sockaddr_utils_test_run(__unused int64_t in, int64_t *out)
{
	test_valid_conversions_to_bytes();
	test_valid_conversions_to_sockaddr();
	test_valid_conversions_to_sockaddr_ctl();
	test_valid_conversions_to_sockaddr_dl();
	test_valid_conversions_to_sockaddr_in();
	test_valid_conversions_to_sockaddr_in6();
	test_valid_conversions_to_sockaddr_inarp();
	test_valid_conversions_to_sockaddr_inifscope();
	test_valid_conversions_to_sockaddr_ndrv();
	test_valid_conversions_to_sockaddr_sys();
	test_valid_conversions_to_sockaddr_un();

	*out = 0;
	return 0;
}

SYSCTL_TEST_REGISTER(sockaddr_utils_test, sockaddr_utils_test_run);

#endif /* DEVELOPMENT || DEBUG */
