/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 16, 2025.
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
#ifdef __cplusplus
extern "C" {
#endif

#ifdef HEIM_FUZZER_INTERNALS
struct heim_fuzz_type_data {
    const char *name;
    unsigned long (*tries)(size_t);
    int (*fuzz)(void **, unsigned long, uint8_t *, size_t);
    void (*freectx)(void *);
};
#endif

typedef const struct heim_fuzz_type_data * heim_fuzz_type_t;

extern const struct heim_fuzz_type_data __heim_fuzz_random;
#define HEIM_FUZZ_RANDOM  (&__heim_fuzz_random)

extern const struct heim_fuzz_type_data __heim_fuzz_bitflip;
#define HEIM_FUZZ_BITFLIP  (&__heim_fuzz_bitflip)

extern const struct heim_fuzz_type_data __heim_fuzz_byteflip;
#define HEIM_FUZZ_BYTEFLIP  (&__heim_fuzz_byteflip)

extern const struct heim_fuzz_type_data __heim_fuzz_shortflip;
#define HEIM_FUZZ_SHORTFLIP  (&__heim_fuzz_shortflip)

extern const struct heim_fuzz_type_data __heim_fuzz_wordflip;
#define HEIM_FUZZ_WORDFLIP  (&__heim_fuzz_wordflip)

extern const struct heim_fuzz_type_data __heim_fuzz_interesting8;
#define HEIM_FUZZ_INTERESTING8  (&__heim_fuzz_interesting8)

extern const struct heim_fuzz_type_data __heim_fuzz_interesting16;
#define HEIM_FUZZ_INTERESTING16  (&__heim_fuzz_interesting16)

extern const struct heim_fuzz_type_data __heim_fuzz_interesting32;
#define HEIM_FUZZ_INTERESTING32  (&__heim_fuzz_interesting32)

/* part of libheimdal-asn1 */
extern const struct heim_fuzz_type_data __heim_fuzz_asn1;
#define HEIM_FUZZ_ASN1  (&__heim_fuzz_asn1)

const char *	heim_fuzzer_name(heim_fuzz_type_t);
int		heim_fuzzer(heim_fuzz_type_t, void **, unsigned long, uint8_t *, size_t);
void		heim_fuzzer_free(heim_fuzz_type_t, void *);

unsigned long	heim_fuzzer_tries(heim_fuzz_type_t, size_t);

#ifdef __cplusplus
}
#endif
