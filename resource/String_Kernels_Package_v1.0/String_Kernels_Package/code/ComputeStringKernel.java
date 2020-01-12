/*
 This code computes a chosen type of string kernel between the texts 
 given in an input file.
 
 Copyright (C) 2015  Radu Tudor Ionescu, Marius Popescu
 
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

public class ComputeStringKernel
{
    private String kernelType;
    private int ngramMinLength;
    private int ngramMaxLength;
    
    public ComputeStringKernel(String kernelType, int ngramMinLength, int ngramMaxLength)
    {
        this.kernelType = kernelType;
        this.ngramMinLength = ngramMinLength;
        this.ngramMaxLength = ngramMaxLength;
    }

    public void makeKernelMatrixForInputSamples(String inputFile, String outputFile)
    {
        String text;
        int n = 0;

        Vector<String> samples = new Vector<String>();

        BufferedReader in;

        try
        {
            in = new BufferedReader(new InputStreamReader(new FileInputStream(inputFile), "UTF-8"));
            text = in.readLine();
            
            while (text != null)
            {
               text = text.trim();
               text = text.toLowerCase();
               text = text.replaceAll("\\s+", " ");
               samples.add(text);
               n++;
               text = in.readLine();
            }
            in.close();

            System.out.println("Loaded " + n + " samples from " + inputFile);

        }
        catch (IOException e)
        {
            System.out.println(e);
            System.exit(1);
        }
        
        System.out.println("Computing the " + kernelType + " kernel based on " + ngramMinLength + "-" + ngramMaxLength + "-grams ...");

        if (kernelType.equals("intersection"))
        {
            BlendedIntersectionStringKernel intStrK = new BlendedIntersectionStringKernel(ngramMinLength, ngramMaxLength);
        
            long[][] K = intStrK.computeKernelMatrix(samples);
            
            System.out.println("Saving kernel matrix to " + outputFile + " ...");
            writeKernelMatrix(K, outputFile);
        }
        else if (kernelType.equals("presence"))
        {
            BlendedPresenceStringKernel presStrK = new BlendedPresenceStringKernel(ngramMinLength, ngramMaxLength);
            
            long[][] K = presStrK.computeKernelMatrix(samples);
            
            System.out.println("Saving kernel matrix to " + outputFile + " ...");
            writeKernelMatrix(K, outputFile);
        }
        else if (kernelType.equals("spectrum"))
        {
            BlendedSpectrumStringKernel presStrK = new BlendedSpectrumStringKernel(ngramMinLength, ngramMaxLength);
            
            long[][] K = presStrK.computeKernelMatrix(samples);
            
            System.out.println("Saving kernel matrix to " + outputFile + " ...");
            writeKernelMatrix(K, outputFile);
        }
        else
        {
            System.out.println("Unknown kernel type:");
            System.out.println("\t<kernelType> can be \"intersection\", \"presence\" or \"spectrum\".");
        }
    }
	
	public void makeKernelMatrixForInputSamples_two(String inputFile_1, String inputFile_2, String outputFile)
    {
        String text;
        int n = 0;

        Vector<String> samples = new Vector<String>();

        BufferedReader in;

        try
        {
            in = new BufferedReader(new InputStreamReader(new FileInputStream(inputFile_1), "UTF-8"));
            text = in.readLine();
            
            while (text != null)
            {
               text = text.trim();
               text = text.toLowerCase();
               text = text.replaceAll("\\s+", " ");
               samples.add(text);
               n++;
               text = in.readLine();
            }
            in.close();

            System.out.println("Loaded " + n + " samples from " + inputFile_1);

        }
        catch (IOException e)
        {
            System.out.println(e);
            System.exit(1);
        }
        
        n = 0;

        Vector<String> samples_1 = new Vector<String>();

        BufferedReader in_1;

        try
        {
            in_1 = new BufferedReader(new InputStreamReader(new FileInputStream(inputFile_2), "UTF-8"));
            text = in_1.readLine();
            
            while (text != null)
            {
               text = text.trim();
               text = text.toLowerCase();
               text = text.replaceAll("\\s+", " ");
               samples_1.add(text);
               n++;
               text = in_1.readLine();
            }
            in_1.close();

            System.out.println("Loaded " + n + " samples from " + inputFile_2);

        }
        catch (IOException e)
        {
            System.out.println(e);
            System.exit(1);
        }
        
        System.out.println("Computing the " + kernelType + " kernel based on " + ngramMinLength + "-" + ngramMaxLength + "-grams ...");

        if (kernelType.equals("intersection"))
        {
            BlendedIntersectionStringKernel intStrK = new BlendedIntersectionStringKernel(ngramMinLength, ngramMaxLength);
        
            long[][] K = intStrK.computeKernelMatrix(samples, samples_1);
            
            System.out.println("Saving kernel matrix to " + outputFile + " ...");
            writeKernelMatrix(K, outputFile);
        }
        else if (kernelType.equals("presence"))
        {
            BlendedPresenceStringKernel presStrK = new BlendedPresenceStringKernel(ngramMinLength, ngramMaxLength);
            
            long[][] K = presStrK.computeKernelMatrix(samples, samples_1);
            
            System.out.println("Saving kernel matrix to " + outputFile + " ...");
            writeKernelMatrix(K, outputFile);
        }
        else if (kernelType.equals("spectrum"))
        {
            BlendedSpectrumStringKernel presStrK = new BlendedSpectrumStringKernel(ngramMinLength, ngramMaxLength);
            
            long[][] K = presStrK.computeKernelMatrix(samples, samples_1);
            
            System.out.println("Saving kernel matrix to " + outputFile + " ...");
            writeKernelMatrix(K, outputFile);
        }
        else
        {
            System.out.println("Unknown kernel type:");
            System.out.println("\t<kernelType> can be \"intersection\", \"presence\" or \"spectrum\".");
        }
    }
	
    public void writeKernelMatrix(long[][] K, String outputFile)
    {
        BufferedWriter out;
        try
        {
            out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputFile)));
            
            for (int i = 0; i < K.length; i++)
            {
                for (int j = 0; j < K[i].length; j++)
                {
                    out.write(K[i][j] + " ");
                }
                out.write("\n");
            }
            out.close();

        }
        catch (IOException e)
        {
            System.out.println(e);
            System.exit(1);
        }
    }

    public static void main(String[] args)
    {	
        System.out.println(args);
        System.out.println(args.length);
        if (args.length > 6)
        {
            System.out.println("Usage: java ComputeStringKernel <kernelType> <ngramMinLength> <ngramMaxLength> <inputFile> <outputFile>");
            System.out.println("Parameters:");
            System.out.println("\t<kernelType> can be \"intersection\", \"presence\" or \"spectrum\";");
            System.out.println("\t<ngramMinLength> and <ngramMaxLength> specify the range of n-grams;");
            System.out.println("\t<inputFile> is an input text file that contains a set of text samples (one sample per row);");
            System.out.println("\t<outputFile> is the generated output file with the pairwise kernel matrix.");
        }
        else
        {
            ComputeStringKernel CSK = new ComputeStringKernel(args[0], Integer.parseInt(args[1]), Integer.parseInt(args[2]));
            if (args.length == 5)
			{
				CSK.makeKernelMatrixForInputSamples(args[3], args[4]);
			}
			else
			{
				CSK.makeKernelMatrixForInputSamples_two(args[3], args[4], args[5]);
			}
        }
    }
}
