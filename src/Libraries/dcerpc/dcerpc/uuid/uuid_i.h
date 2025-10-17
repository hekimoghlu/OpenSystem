/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 14, 2023.
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
**
**  NAME:
**
**      uuidp.h
**
**  FACILITY:
**
**      UUID
**
**  ABSTRACT:
**
**      Interface Definitions for UUID subroutines used by
**      the uuid module
**
**
*/

#ifndef _UUIDP_H
#define _UUIDP_H	1

/*
 * Max size of a uuid string: tttttttt-tttt-cccc-cccc-nnnnnnnnnnnn
 * Note: this includes the implied '\0'
 */
#define UUID_C_UUID_STRING_MAX          37

/*
 * defines for time calculations
 */
#ifndef UUID_C_100NS_PER_SEC
#define UUID_C_100NS_PER_SEC            10000000
#endif

#ifndef UUID_C_100NS_PER_USEC
#define UUID_C_100NS_PER_USEC           10
#endif



/*
 * UADD_UVLW_2_UVLW - macro to add two unsigned 64-bit long integers
 *                      (ie. add two unsigned 'very' long words)
 *
 * Important note: It is important that this macro accommodate (and it does)
 *                 invocations where one of the addends is also the sum.
 *
 * This macro was snarfed from the DTSS group and was originally:
 *
 * UTCadd - macro to add two UTC times
 *
 * add lo and high order longword separately, using sign bits of the low-order
 * longwords to determine carry.  sign bits are tested before addition in two
 * cases - where sign bits match. when the addend sign bits differ the sign of
 * the result is also tested:
 *
 *        sign            sign
 *      addend 1        addend 2        carry?
 *
 *          1               1            TRUE
 *          1               0            TRUE if sign of sum clear
 *          0               1            TRUE if sign of sum clear
 *          0               0            FALSE
 */
#define UADD_UVLW_2_UVLW(add1, add2, sum)                               \
    if (!(((add1)->lo&0x80000000UL) ^ ((add2)->lo&0x80000000UL)))           \
    {                                                                   \
        if (((add1)->lo&0x80000000UL))                                    \
        {                                                               \
            (sum)->lo = (add1)->lo + (add2)->lo ;                       \
            (sum)->hi = (add1)->hi + (add2)->hi+1 ;                     \
        }                                                               \
        else                                                            \
        {                                                               \
            (sum)->lo  = (add1)->lo + (add2)->lo ;                      \
            (sum)->hi = (add1)->hi + (add2)->hi ;                       \
        }                                                               \
    }                                                                   \
    else                                                                \
    {                                                                   \
        (sum)->lo = (add1)->lo + (add2)->lo ;                           \
        (sum)->hi = (add1)->hi + (add2)->hi ;                           \
        if (!((sum)->lo&0x80000000UL))                                    \
            (sum)->hi++ ;                                               \
    }


/*
 * UADD_ULW_2_UVLW - macro to add a 32-bit unsigned integer to
 *                   a 64-bit unsigned integer
 *
 * Note: see the UADD_UVLW_2_UVLW() macro
 *
 */
#define UADD_ULW_2_UVLW(add1, add2, sum)                                \
{                                                                       \
    (sum)->hi = (add2)->hi;                                             \
    if ((*add1) & (add2)->lo & 0x80000000UL)                              \
    {                                                                   \
        (sum)->lo = (*add1) + (add2)->lo;                               \
        (sum)->hi++;                                                    \
    }                                                                   \
    else                                                                \
    {                                                                   \
        (sum)->lo = (*add1) + (add2)->lo;                               \
        if (!((sum)->lo & 0x80000000UL))                                  \
        {                                                               \
            (sum)->hi++;                                                \
        }                                                               \
    }                                                                   \
}

/*
 * UADD_UW_2_UVLW - macro to add a 16-bit unsigned integer to
 *                   a 64-bit unsigned integer
 *
 * Note: see the UADD_UVLW_2_UVLW() macro
 *
 */
#define UADD_UW_2_UVLW(add1, add2, sum)                                 \
{                                                                       \
    (sum)->hi = (add2)->hi;                                             \
    if ((add2)->lo & 0x80000000UL)                                        \
    {                                                                   \
        (sum)->lo = (*add1) + (add2)->lo;                               \
        if (!((sum)->lo & 0x80000000UL))                                  \
        {                                                               \
            (sum)->hi++;                                                \
        }                                                               \
    }                                                                   \
    else                                                                \
    {                                                                   \
        (sum)->lo = (*add1) + (add2)->lo;                               \
    }                                                                   \
}



/*
 * a macro to set *status uuid_s_coding_error
 */
#ifdef  CODING_ERROR
#undef  CODING_ERROR
#endif

#ifdef  DEBUG
#define CODING_ERROR(status)        *(status) = uuid_s_coding_error
#else
#define CODING_ERROR(status)
#endif

typedef struct
{
    char eaddr[6];      /* 6 bytes of ethernet hardware address */
} uuid_address_t, *uuid_address_p_t;

typedef struct
{
    unsigned32  lo;
    unsigned32  hi;
} uuid_time_t, *uuid_time_p_t;

typedef struct
{
    unsigned32  lo;
    unsigned32  hi;
} unsigned64_t, *unsigned64_p_t;

/*
 * U U I D _ _ U E M U L
 *
 * 32-bit unsigned * 32-bit unsigned multiply -> 64-bit result
 */
 void uuid__uemul (
        unsigned32           /*u*/,
        unsigned32           /*v*/,
        unsigned64_t        * /*prodPtr*/
    );

/*
 * U U I D _ _ R E A D _ C L O C K
 */
 unsigned16 uuid__read_clock ( void );

/*
 * U U I D _ _ W R I T E _ C L O C K
 */
 void uuid__write_clock ( unsigned16 /*time*/ );

/*
 * U U I D _ _ G E T _ O S _ P I D
 *
 * Get the process id
 */
 unsigned32 uuid__get_os_pid ( void );

/*
 * U U I D _ _ G E T _ O S _ T I M E
 *
 * Get OS time
 */
 void uuid__get_os_time ( uuid_time_t * /*uuid_time*/);

/*
 * U U I D _ _ G E T _ O S _ A D D R E S S
 *
 * Get ethernet hardware address from the OS
 */
 void uuid__get_os_address (
        uuid_address_t          * /*address*/,
        unsigned32              * /*st*/
    );

#endif /* _UUIDP_H */
