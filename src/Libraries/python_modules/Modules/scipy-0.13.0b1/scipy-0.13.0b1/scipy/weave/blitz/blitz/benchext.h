/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 28, 2025.
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
#ifndef BZ_BENCHEXT_H
#define BZ_BENCHEXT_H

#ifndef BZ_MATRIX_H
 #include <blitz/matrix.h>
#endif

#ifndef BZ_TIMER_H
 #include <blitz/timer.h>
#endif

#include <math.h>

// NEEDS_WORK: replace use of const char* with <string>, once standard
// library is widely supported.

BZ_NAMESPACE(blitz)

// Declaration of class BenchmarkExt<T>
// The template parameter T is the parameter type which is varied in
// the benchmark.  Typically T will be an unsigned, and will represent
// the length of a vector, size of an array, etc.

template<typename P_parameter = unsigned>
class BenchmarkExt {

public:
    typedef P_parameter T_parameter;

    BenchmarkExt(const char* description, int numImplementations);

    ~BenchmarkExt();

    void setNumParameters(int numParameters);
    void setParameterVector(Vector<T_parameter> parms);
    void setParameterDescription(const char* string);
    void setIterations(Vector<long> iters);
    void setFlopsPerIteration(Vector<double> flopsPerIteration);
    void setRateDescription(const char* string);

    void beginBenchmarking();

    void beginImplementation(const char* description);

    bool doneImplementationBenchmark() const;

    T_parameter getParameter() const;
    long        getIterations() const;

    inline void start();
    inline void stop();

    void startOverhead();
    void stopOverhead();

    void endImplementation();

    void endBenchmarking();
 
    double getMflops(unsigned implementation, unsigned parameterNum) const;

    void saveMatlabGraph(const char* filename, const char* graphType="semilogx") const;

protected:
    BenchmarkExt(const BenchmarkExt<P_parameter>&) { }
    void operator=(const BenchmarkExt<P_parameter>&) { }

    enum { initializing, benchmarking, benchmarkingImplementation, 
       running, runningOverhead, done } state_;

    unsigned numImplementations_;
    unsigned implementationNumber_;

    const char* description_;
    Vector<const char*> implementationDescriptions_;

    Matrix<double,RowMajor> times_;       // Elapsed time

    Vector<T_parameter> parameters_;
    Vector<long> iterations_;
    Vector<double> flopsPerIteration_;

    Timer timer_;
    Timer overheadTimer_;

    const char* parameterDescription_;
    const char* rateDescription_;

    unsigned numParameters_;
    unsigned parameterNumber_;
};

BZ_NAMESPACE_END

#include <blitz/benchext.cc>  

#endif // BZ_BENCHEXT_H
