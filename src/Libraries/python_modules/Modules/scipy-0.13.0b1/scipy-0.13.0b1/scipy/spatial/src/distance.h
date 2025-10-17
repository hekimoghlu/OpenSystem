/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 23, 2023.
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
#ifndef _CPY_DISTANCE_H
#define _CPY_DISTANCE_H

void dist_to_squareform_from_vector(double *M, const double *v, int n);
void dist_to_vector_from_squareform(const double *M, double *v, int n);
void pdist_euclidean(const double *X, double *dm, int m, int n);
void pdist_seuclidean(const double *X,
		      const double *var, double *dm, int m, int n);
void pdist_mahalanobis(const double *X, const double *covinv,
		       double *dm, int m, int n);
void pdist_bray_curtis(const double *X, double *dm, int m, int n);
void pdist_canberra(const double *X, double *dm, int m, int n);
void pdist_hamming(const double *X, double *dm, int m, int n);
void pdist_hamming_bool(const char *X, double *dm, int m, int n);
void pdist_city_block(const double *X, double *dm, int m, int n);
void pdist_cosine(const double *X, double *dm, int m, int n, const double *norms);
void pdist_chebyshev(const double *X, double *dm, int m, int n);
void pdist_jaccard(const double *X, double *dm, int m, int n);
void pdist_jaccard_bool(const char *X, double *dm, int m, int n);
void pdist_kulsinski_bool(const char *X, double *dm, int m, int n);
void pdist_minkowski(const double *X, double *dm, int m, int n, double p);
void pdist_weighted_minkowski(const double *X, double *dm, int m, int n, double p, const double *w);
void pdist_yule_bool(const char *X, double *dm, int m, int n);
void pdist_matching_bool(const char *X, double *dm, int m, int n);
void pdist_dice_bool(const char *X, double *dm, int m, int n);
void pdist_rogerstanimoto_bool(const char *X, double *dm, int m, int n);
void pdist_russellrao_bool(const char *X, double *dm, int m, int n);
void pdist_sokalmichener_bool(const char *X, double *dm, int m, int n);
void pdist_sokalsneath_bool(const char *X, double *dm, int m, int n);

void cdist_euclidean(const double *XA, const double *XB, double *dm, int mA, int mB, int n);
void cdist_mahalanobis(const double *XA, const double *XB,
		       const double *covinv,
		       double *dm, int mA, int mB, int n);
void cdist_bray_curtis(const double *XA, const double *XB,
		       double *dm, int mA, int mB, int n);
void cdist_canberra(const double *XA,
		    const double *XB, double *dm, int mA, int mB, int n);
void cdist_hamming(const double *XA,
		   const double *XB, double *dm, int mA, int mB, int n);
void cdist_hamming_bool(const char *XA,
			const char *XB, double *dm,
			int mA, int mB, int n);
void cdist_jaccard(const double *XA,
		   const double *XB, double *dm, int mA, int mB, int n);
void cdist_jaccard_bool(const char *XA,
			const char *XB, double *dm, int mA, int mB, int n);
void cdist_chebyshev(const double *XA,
		     const double *XB, double *dm, int mA, int mB, int n);
void cdist_cosine(const double *XA,
		  const double *XB, double *dm, int mA, int mB, int n,
		  const double *normsA, const double *normsB);
void cdist_seuclidean(const double *XA,
		      const double *XB,
		      const double *var,
		      double *dm, int mA, int mB, int n);
void cdist_city_block(const double *XA, const double *XB, double *dm,
		      int mA, int mB, int n);
void cdist_minkowski(const double *XA, const double *XB, double *dm,
		     int mA, int mB, int n, double p);
void cdist_weighted_minkowski(const double *XA, const double *XB, double *dm,
			      int mA, int mB, int n, double p, const double *w);
void cdist_yule_bool(const char *XA, const char *XB, double *dm,
		     int mA, int mB, int n);
void cdist_matching_bool(const char *XA, const char *XB, double *dm,
			 int mA, int mB, int n);
void cdist_dice_bool(const char *XA, const char *XB, double *dm,
		     int mA, int mB, int n);
void cdist_rogerstanimoto_bool(const char *XA, const char *XB, double *dm,
			       int mA, int mB, int n);
void cdist_russellrao_bool(const char *XA, const char *XB, double *dm,
			   int mA, int mB, int n);
void cdist_kulsinski_bool(const char *XA, const char *XB, double *dm,
			  int mA, int mB, int n);
void cdist_sokalsneath_bool(const char *XA, const char *XB, double *dm,
			    int mA, int mB, int n);
void cdist_sokalmichener_bool(const char *XA, const char *XB, double *dm,
			      int mA, int mB, int n);

#endif
