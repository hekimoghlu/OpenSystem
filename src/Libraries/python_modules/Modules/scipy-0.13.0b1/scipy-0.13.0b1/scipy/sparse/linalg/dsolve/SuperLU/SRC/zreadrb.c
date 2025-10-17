/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 6, 2023.
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
#include "slu_zdefs.h"


/*! \brief Eat up the rest of the current line */
static int zDumpLine(FILE *fp)
{
    register int c;
    while ((c = fgetc(fp)) != '\n') ;
    return 0;
}

static int zParseIntFormat(char *buf, int *num, int *size)
{
    char *tmp;

    tmp = buf;
    while (*tmp++ != '(') ;
    sscanf(tmp, "%d", num);
    while (*tmp != 'I' && *tmp != 'i') ++tmp;
    ++tmp;
    sscanf(tmp, "%d", size);
    return 0;
}

static int zParseFloatFormat(char *buf, int *num, int *size)
{
    char *tmp, *period;

    tmp = buf;
    while (*tmp++ != '(') ;
    *num = atoi(tmp); /*sscanf(tmp, "%d", num);*/
    while (*tmp != 'E' && *tmp != 'e' && *tmp != 'D' && *tmp != 'd'
           && *tmp != 'F' && *tmp != 'f') {
        /* May find kP before nE/nD/nF, like (1P6F13.6). In this case the
           num picked up refers to P, which should be skipped. */
        if (*tmp=='p' || *tmp=='P') {
           ++tmp;
           *num = atoi(tmp); /*sscanf(tmp, "%d", num);*/
        } else {
           ++tmp;
        }
    }
    ++tmp;
    period = tmp;
    while (*period != '.' && *period != ')') ++period ;
    *period = '\0';
    *size = atoi(tmp); /*sscanf(tmp, "%2d", size);*/

    return 0;
}

static int ReadVector(FILE *fp, int n, int *where, int perline, int persize)
{
    register int i, j, item;
    char tmp, buf[100];

    i = 0;
    while (i < n) {
        fgets(buf, 100, fp);    /* read a line at a time */
        for (j=0; j<perline && i<n; j++) {
            tmp = buf[(j+1)*persize];     /* save the char at that place */
            buf[(j+1)*persize] = 0;       /* null terminate */
            item = atoi(&buf[j*persize]); 
            buf[(j+1)*persize] = tmp;     /* recover the char at that place */
            where[i++] = item - 1;
        }
    }

    return 0;
}

/*! \brief Read complex numbers as pairs of (real, imaginary) */
static int zReadValues(FILE *fp, int n, doublecomplex *destination, int perline, int persize)
{
    register int i, j, k, s, pair;
    register double realpart;
    char tmp, buf[100];
    
    i = pair = 0;
    while (i < n) {
	fgets(buf, 100, fp);    /* read a line at a time */
	for (j=0; j<perline && i<n; j++) {
	    tmp = buf[(j+1)*persize];     /* save the char at that place */
	    buf[(j+1)*persize] = 0;       /* null terminate */
	    s = j*persize;
	    for (k = 0; k < persize; ++k) /* No D_ format in C */
		if ( buf[s+k] == 'D' || buf[s+k] == 'd' ) buf[s+k] = 'E';
	    if ( pair == 0 ) {
	  	/* The value is real part */
		realpart = atof(&buf[s]);
		pair = 1;
	    } else {
		/* The value is imaginary part */
	        destination[i].r = realpart;
		destination[i++].i = atof(&buf[s]);
		pair = 0;
	    }
	    buf[(j+1)*persize] = tmp;     /* recover the char at that place */
	}
    }

    return 0;
}


void
zreadrb(int *nrow, int *ncol, int *nonz,
        doublecomplex **nzval, int **rowind, int **colptr)
{

    register int i, numer_lines = 0;
    int tmp, colnum, colsize, rownum, rowsize, valnum, valsize;
    char buf[100], type[4];
    FILE *fp;

    fp = stdin;

    /* Line 1 */
    fgets(buf, 100, fp);
    fputs(buf, stdout);

    /* Line 2 */
    for (i=0; i<4; i++) {
        fscanf(fp, "%14c", buf); buf[14] = 0;
        sscanf(buf, "%d", &tmp);
        if (i == 3) numer_lines = tmp;
    }
    zDumpLine(fp);

    /* Line 3 */
    fscanf(fp, "%3c", type);
    fscanf(fp, "%11c", buf); /* pad */
    type[3] = 0;
#ifdef DEBUG
    printf("Matrix type %s\n", type);
#endif

    fscanf(fp, "%14c", buf); sscanf(buf, "%d", nrow);
    fscanf(fp, "%14c", buf); sscanf(buf, "%d", ncol);
    fscanf(fp, "%14c", buf); sscanf(buf, "%d", nonz);
    fscanf(fp, "%14c", buf); sscanf(buf, "%d", &tmp);

    if (tmp != 0)
        printf("This is not an assembled matrix!\n");
    if (*nrow != *ncol)
        printf("Matrix is not square.\n");
    zDumpLine(fp);

    /* Allocate storage for the three arrays ( nzval, rowind, colptr ) */
    zallocateA(*ncol, *nonz, nzval, rowind, colptr);

    /* Line 4: format statement */
    fscanf(fp, "%16c", buf);
    zParseIntFormat(buf, &colnum, &colsize);
    fscanf(fp, "%16c", buf);
    zParseIntFormat(buf, &rownum, &rowsize);
    fscanf(fp, "%20c", buf);
    zParseFloatFormat(buf, &valnum, &valsize);
    zDumpLine(fp);

#ifdef DEBUG
    printf("%d rows, %d nonzeros\n", *nrow, *nonz);
    printf("colnum %d, colsize %d\n", colnum, colsize);
    printf("rownum %d, rowsize %d\n", rownum, rowsize);
    printf("valnum %d, valsize %d\n", valnum, valsize);
#endif

    ReadVector(fp, *ncol+1, *colptr, colnum, colsize);
    ReadVector(fp, *nonz, *rowind, rownum, rowsize);
    if ( numer_lines ) {
        zReadValues(fp, *nonz, *nzval, valnum, valsize);
    }

    fclose(fp);
}
