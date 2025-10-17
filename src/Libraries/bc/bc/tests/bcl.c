/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 12, 2021.
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
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

#include <bcl.h>

/**
 * Takes an error code and aborts if it actually is an error.
 * @param e  The error code.
 */
static void
err(BclError e)
{
	if (e != BCL_ERROR_NONE) abort();
}

int
main(void)
{
	BclError e;
	BclContext ctxt;
	size_t scale;
	BclNumber n, n2, n3, n4, n5, n6, n7;
	char* res;
	BclBigDig b = 0;

	e = bcl_start();
	err(e);

	// We do this twice to test the reference counting code.
	e = bcl_init();
	err(e);
	e = bcl_init();
	err(e);

	// If bcl is set to abort on fatal error, that is a bug because it should
	// default to off.
	if (bcl_abortOnFatalError()) err(BCL_ERROR_FATAL_UNKNOWN_ERR);

	bcl_setAbortOnFatalError(true);

	// Now it *should* be set.
	if (!bcl_abortOnFatalError()) err(BCL_ERROR_FATAL_UNKNOWN_ERR);

	// We do this twice to test the context stack.
	ctxt = bcl_ctxt_create();
	bcl_pushContext(ctxt);
	ctxt = bcl_ctxt_create();
	bcl_pushContext(ctxt);

	// Ensure that the scale is properly set.
	scale = 10;
	bcl_ctxt_setScale(ctxt, scale);
	scale = bcl_ctxt_scale(ctxt);
	if (scale != 10) err(BCL_ERROR_FATAL_UNKNOWN_ERR);

	scale = 16;
	bcl_ctxt_setIbase(ctxt, scale);
	scale = bcl_ctxt_ibase(ctxt);
	if (scale != 16) err(BCL_ERROR_FATAL_UNKNOWN_ERR);

	// Now the obase.
	bcl_ctxt_setObase(ctxt, scale);
	scale = bcl_ctxt_obase(ctxt);
	if (scale != 16) err(BCL_ERROR_FATAL_UNKNOWN_ERR);

	// Set the back for the tests
	bcl_ctxt_setIbase(ctxt, 10);
	scale = bcl_ctxt_ibase(ctxt);
	if (scale != 10) err(BCL_ERROR_FATAL_UNKNOWN_ERR);
	bcl_ctxt_setObase(ctxt, 10);
	scale = bcl_ctxt_obase(ctxt);
	if (scale != 10) err(BCL_ERROR_FATAL_UNKNOWN_ERR);

	// Ensure that creating, duping, and copying works.
	n = bcl_num_create();
	n2 = bcl_dup(n);
	bcl_copy(n, n2);

	// Ensure that parsing works.
	n3 = bcl_parse("2938");
	err(bcl_err(n3));
	n4 = bcl_parse("-28390.9108273");
	err(bcl_err(n4));

	// We also want to be sure that negatives work. This is a special case
	// because bc and dc generate a negative instruction; they don't actually
	// parse numbers as negative.
	if (!bcl_num_neg(n4)) err(BCL_ERROR_FATAL_UNKNOWN_ERR);

	// Add them and check the result.
	n5 = bcl_add_keep(n3, n4);
	err(bcl_err(n5));
	res = bcl_string(n5);
	if (res == NULL) err(BCL_ERROR_FATAL_ALLOC_ERR);
	if (strcmp(res, "-25452.9108273")) err(BCL_ERROR_FATAL_UNKNOWN_ERR);

	// We want to ensure all memory gets freed because we run this under
	// Valgrind.
	free(res);

	// Add them and check the result.
	n3 = bcl_add(n3, n4);
	err(bcl_err(n3));
	res = bcl_string_keep(n3);
	if (res == NULL) err(BCL_ERROR_FATAL_ALLOC_ERR);
	if (strcmp(res, "-25452.9108273")) err(BCL_ERROR_FATAL_UNKNOWN_ERR);

	// We want to ensure all memory gets freed because we run this under
	// Valgrind.
	free(res);

	// Ensure that divmod, a special case, works.
	n4 = bcl_parse("8937458902.2890347");
	err(bcl_err(n4));
	e = bcl_divmod_keep(n4, n3, &n5, &n6);
	err(e);

	res = bcl_string(n5);
	if (strcmp(res, "-351137.0060159482")) err(BCL_ERROR_FATAL_UNKNOWN_ERR);
	free(res);

	res = bcl_string(n6);
	if (strcmp(res, ".00000152374405414")) err(BCL_ERROR_FATAL_UNKNOWN_ERR);
	free(res);

	// Ensure that divmod, a special case, works.
	n4 = bcl_parse("8937458902.2890347");
	err(bcl_err(n4));
	e = bcl_divmod(bcl_dup(n4), n3, &n5, &n6);
	err(e);

	res = bcl_string(n5);
	if (strcmp(res, "-351137.0060159482")) err(BCL_ERROR_FATAL_UNKNOWN_ERR);
	free(res);

	res = bcl_string(n6);
	if (strcmp(res, ".00000152374405414")) err(BCL_ERROR_FATAL_UNKNOWN_ERR);
	free(res);

	// Ensure that sqrt works. This is also a special case. The reason is
	// because it is a one-argument function. Since all binary operators go
	// through the same code (basically), we can test add and be done. However,
	// sqrt does not, so we want to specifically test it.
	n4 = bcl_sqrt(n4);
	err(bcl_err(n4));

	res = bcl_string(bcl_dup(n4));

	if (strcmp(res, "94538.1346457028")) err(BCL_ERROR_FATAL_UNKNOWN_ERR);

	free(res);

	// We want to check that numbers are properly extended...
	e = bcl_num_setScale(n4, 20);
	err(e);

	res = bcl_string(bcl_dup(n4));

	if (strcmp(res, "94538.13464570280000000000"))
		err(BCL_ERROR_FATAL_UNKNOWN_ERR);

	free(res);

	// ...and truncated.
	e = bcl_num_setScale(n4, 0);
	err(e);

	res = bcl_string(bcl_dup(n4));

	if (strcmp(res, "94538")) err(BCL_ERROR_FATAL_UNKNOWN_ERR);

	free(res);

	// Check conversion to hardware integers...
	e = bcl_bigdig(n4, &b);
	err(e);

	if (b != 94538) err(BCL_ERROR_FATAL_UNKNOWN_ERR);

	// ...and back.
	n4 = bcl_bigdig2num(b);
	err(bcl_err(n4));

	res = bcl_string(bcl_dup(n4));

	if (strcmp(res, "94538")) err(BCL_ERROR_FATAL_UNKNOWN_ERR);

	free(res);

	// Check rand.
	n4 = bcl_frand(10);
	err(bcl_err(n4));

	// Check that no asserts fire in shifting.
	n4 = bcl_lshift(n4, bcl_bigdig2num(10));
	err(bcl_err(n4));

	// Repeat.
	n3 = bcl_irand(n4);
	err(bcl_err(n3));

	// Repeat.
	n2 = bcl_ifrand_keep(n3, 10);
	err(bcl_err(n2));

	// Repeat.
	n2 = bcl_ifrand(bcl_dup(n3), 10);
	err(bcl_err(n2));

	// Still checking asserts.
	e = bcl_rand_seedWithNum_keep(n3);
	err(e);

	// Still checking asserts.
	e = bcl_rand_seedWithNum(n3);
	err(e);

	// Still checking asserts.
	n4 = bcl_rand_seed2num();
	err(bcl_err(n4));

	// Finally, check modexp, yet another special case.
	n5 = bcl_parse("10");
	err(bcl_err(n5));

	n6 = bcl_modexp_keep(n5, n5, n5);
	err(bcl_err(n6));

	n7 = bcl_modexp(bcl_dup(n5), bcl_dup(n5), bcl_dup(n5));
	err(bcl_err(n7));

	// Clean up.
	bcl_num_free(n);

	// Test leading zeroes.
	if (bcl_leadingZeroes()) err(BCL_ERROR_FATAL_UNKNOWN_ERR);

	n = bcl_parse("0.01");
	err(bcl_err(n));

	n2 = bcl_parse("-0.01");
	err(bcl_err(n2));

	n3 = bcl_parse("1.01");
	err(bcl_err(n3));

	n4 = bcl_parse("-1.01");
	err(bcl_err(n4));

	res = bcl_string_keep(n);
	if (strcmp(res, ".01")) err(BCL_ERROR_FATAL_UNKNOWN_ERR);

	free(res);

	res = bcl_string(bcl_dup(n));
	if (strcmp(res, ".01")) err(BCL_ERROR_FATAL_UNKNOWN_ERR);

	free(res);

	res = bcl_string(bcl_dup(n2));
	if (strcmp(res, "-.01")) err(BCL_ERROR_FATAL_UNKNOWN_ERR);

	free(res);

	res = bcl_string(bcl_dup(n3));
	if (strcmp(res, "1.01")) err(BCL_ERROR_FATAL_UNKNOWN_ERR);

	free(res);

	res = bcl_string(bcl_dup(n4));
	if (strcmp(res, "-1.01")) err(BCL_ERROR_FATAL_UNKNOWN_ERR);

	free(res);

	bcl_setLeadingZeroes(true);

	if (!bcl_leadingZeroes()) err(BCL_ERROR_FATAL_UNKNOWN_ERR);

	res = bcl_string(bcl_dup(n));
	if (strcmp(res, "0.01")) err(BCL_ERROR_FATAL_UNKNOWN_ERR);

	free(res);

	res = bcl_string(bcl_dup(n2));
	if (strcmp(res, "-0.01")) err(BCL_ERROR_FATAL_UNKNOWN_ERR);

	free(res);

	res = bcl_string(bcl_dup(n3));
	if (strcmp(res, "1.01")) err(BCL_ERROR_FATAL_UNKNOWN_ERR);

	free(res);

	res = bcl_string(bcl_dup(n4));
	if (strcmp(res, "-1.01")) err(BCL_ERROR_FATAL_UNKNOWN_ERR);

	free(res);

	bcl_setLeadingZeroes(false);

	if (bcl_leadingZeroes()) err(BCL_ERROR_FATAL_UNKNOWN_ERR);

	res = bcl_string(n);
	if (strcmp(res, ".01")) err(BCL_ERROR_FATAL_UNKNOWN_ERR);

	free(res);

	res = bcl_string(n2);
	if (strcmp(res, "-.01")) err(BCL_ERROR_FATAL_UNKNOWN_ERR);

	free(res);

	res = bcl_string(n3);
	if (strcmp(res, "1.01")) err(BCL_ERROR_FATAL_UNKNOWN_ERR);

	free(res);

	res = bcl_string(n4);
	if (strcmp(res, "-1.01")) err(BCL_ERROR_FATAL_UNKNOWN_ERR);

	free(res);

	bcl_ctxt_freeNums(ctxt);

	bcl_gc();

	// We need to pop both contexts and free them.
	bcl_popContext();

	bcl_ctxt_free(ctxt);

	ctxt = bcl_context();

	bcl_popContext();

	bcl_ctxt_free(ctxt);

	// Decrement the reference counter to ensure all is freed.
	bcl_free();

	bcl_free();

	bcl_end();

	return 0;
}
