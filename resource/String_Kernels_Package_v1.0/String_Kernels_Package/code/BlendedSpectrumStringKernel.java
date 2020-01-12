/*
 This code computes the blended spectrum string kernel.
 
 Copyright (C) 2015  Marius Popescu, Radu Tudor Ionescu
 
 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or any
 later version.
 
 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 
 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

import java.io.*;
import java.util.*;

public class BlendedSpectrumStringKernel
{
    private int ngramMinLength;
    private int ngramMaxLength;

    public BlendedSpectrumStringKernel(int ngramMinLength, int ngramMaxLength)
    {
        this.ngramMinLength = ngramMinLength;
        this.ngramMaxLength = ngramMaxLength;
    }

    public BlendedSpectrumStringKernel()
    {
        this(1, 5);
    }
    
    // Computes the blended spectrum kernel among two strings.
    public long computeKernel(String x, String y)
    {
        String ngram;
        Long count;
        long ker;

        Map<String, Long> ngrams = new HashMap<String, Long>();

        for (int l = 0; l < x.length(); l++)
        {
            for (int d = ngramMinLength; d <= ngramMaxLength; d++)
            {
                if (l + d <= x.length())
                {
                    ngram = x.substring(l, l + d);
                    count = ngrams.get(ngram);
                    
                    if (count != null) ngrams.put(ngram, count + 1);
                    else ngrams.put(ngram, 1l);
                }
            }
        }
        
        ker = 0l;
        for (int l = 0; l < y.length(); l++)
        {
            for (int d = ngramMinLength; d <= ngramMaxLength; d++)
            {
                if (l + d <= y.length())
                {
                    ngram = y.substring(l, l + d);
                    count = ngrams.get(ngram);
                    
                    if (count != null)
                    {
                        ker += count;
                    }
                }
            }
        }
        return ker;
    }

    // Computes the blended spectrum kernel matrix for a set of samples.
    public long[][] computeKernelMatrix(Vector<String> samples)
    {
        String sample, ngram;
        Long count;

        long[][] K = new long[samples.size()][samples.size()];
        
        for (int i = 0; i < samples.size(); i++)
        {
            HashMap<String, Long> ngrams = new HashMap<String, Long>();
            sample = samples.get(i);
            
            for (int l = 0; l < sample.length(); l++)
            {
                for (int d = ngramMinLength; d <= ngramMaxLength; d++)
                {
                    if (l + d <= sample.length())
                    {
                        ngram = sample.substring(l, l + d);
                        count = ngrams.get(ngram);
                        
                        if (count != null) ngrams.put(ngram, count + 1);
                        else ngrams.put(ngram, 1l);
                    }
                }
            }
            
            for (int j = i; j < samples.size(); j++)
            {
                K[i][j] = 0l;
                
                HashMap<String, Long> sampleNgrams = (HashMap<String, Long>) ngrams.clone();
                sample = samples.get(j);
                
                for (int l = 0; l < sample.length(); l++)
                {
                    for (int d = ngramMinLength; d <= ngramMaxLength; d++)
                    {
                        if (l + d <= sample.length())
                        {
                            ngram = sample.substring(l, l + d);
                            count = sampleNgrams.get(ngram);
                            
                            if (count != null && count > 0)
                            {
                                K[i][j] += count;
                            }
                        }
                    }
                }
                K[j][i] = K[i][j];
            }
            if ((i + 1) % 100 == 0) System.out.println("Computed kernel to row " + i);
        }

        return K;
    }

    // Computes the blended spectrum kernel matrix among two sets of samples.
    public long[][] computeKernelMatrix(Vector<String> samples1, Vector<String> samples2)
    {
        String sample, ngram;
        Long count;

        long[][] K = new long[samples1.size()][samples2.size()];

        for (int i = 0; i < samples1.size(); i++)
        {
            HashMap<String, Long> ngrams = new HashMap<String, Long>();
            sample = samples1.get(i);
            
            for (int l = 0; l < sample.length(); l++)
            {
                for (int d = ngramMinLength; d <= ngramMaxLength; d++)
                {
                    if (l + d <= sample.length())
                    {
                        ngram = sample.substring(l, l + d);
                        count = ngrams.get(ngram);
                        
                        if (count != null) ngrams.put(ngram, count + 1);
                        else ngrams.put(ngram, 1l);
                    }
                }
            }

            for (int j = 0; j < samples2.size(); j++)
            {
                K[i][j] = 0l;
                
                HashMap<String, Long> sampleNgrams = (HashMap<String, Long>) ngrams.clone();
                sample = samples2.get(j);
                
                for (int l = 0; l < sample.length(); l++)
                {
                    for (int d = ngramMinLength; d <= ngramMaxLength; d++)
                    {
                        if (l + d <= sample.length())
                        {
                            ngram = sample.substring(l, l + d);
                            count = sampleNgrams.get(ngram);
                            
                            if (count != null && count > 0)
                            {
                                K[i][j] += count;
                            }
                        }
                    }
                }

            }
            if ((i + 1) % 100 == 0) System.out.println("Computed kernel to row " + i);
        }

        return K;
    }
}
