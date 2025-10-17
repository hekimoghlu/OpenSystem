/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 7, 2025.
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

#ifndef _NPY_PRIVATE__DATETIME_BUSDAYDEF_H_
#define _NPY_PRIVATE__DATETIME_BUSDAYDEF_H_

/*
 * A list of holidays, which should be sorted, not contain any
 * duplicates or NaTs, and not include any days already excluded
 * by the associated weekmask.
 *
 * The data is manually managed with PyArray_malloc/PyArray_free.
 */
typedef struct {
    npy_datetime *begin, *end;
} npy_holidayslist;

/*
 * This object encapsulates a weekmask and normalized holidays list,
 * so that the business day API can use this data without having
 * to normalize it repeatedly. All the data of this object is private
 * and cannot be modified from Python. Copies are made when giving
 * the weekmask and holidays data to Python code.
 */
typedef struct {
    PyObject_HEAD
    npy_holidayslist holidays;
    int busdays_in_weekmask;
    npy_bool weekmask[7];
} NpyBusDayCalendar;

#ifdef NPY_ENABLE_SEPARATE_COMPILATION
extern NPY_NO_EXPORT PyTypeObject NpyBusDayCalendar_Type;
#else
NPY_NO_EXPORT PyTypeObject NpyBusDayCalendar_Type;
#endif


/*
 * Converts a Python input into a 7-element weekmask, where 0 means
 * weekend and 1 means business day.
 */
NPY_NO_EXPORT int
PyArray_WeekMaskConverter(PyObject *weekmask_in, npy_bool *weekmask);

/*
 * Sorts the the array of dates provided in place and removes
 * NaT, duplicates and any date which is already excluded on account
 * of the weekmask.
 *
 * Returns the number of dates left after removing weekmask-excluded
 * dates.
 */
NPY_NO_EXPORT void
normalize_holidays_list(npy_holidayslist *holidays, npy_bool *weekmask);

/*
 * Converts a Python input into a non-normalized list of holidays.
 *
 * IMPORTANT: This function can't do the normalization, because it doesn't
 *            know the weekmask. You must call 'normalize_holiday_list'
 *            on the result before using it.
 */
NPY_NO_EXPORT int
PyArray_HolidaysConverter(PyObject *dates_in, npy_holidayslist *holidays);



#endif
