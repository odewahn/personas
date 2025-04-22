The following selection of text is a transcript of a recording of a person speaking:

{{content}}

I'm trying to create a system that can mimic this person's speaking style.  

I want you take each full sentence (ignore partial sentences) and create a generic version. The goal of the generic version is to really highlight the differences in the spoken version, so try to make it really bland.  Produce an output like this:

<thinking>
  <sample>
    <boring>
      ... generic version 1...
    </boring>
    <after>
      ... original sentence 1...
    </after> 
  </sample>
  <sample>
    <boring>
      ... generic version 2...
    </boring>
    <after>
      ... original sentence 2...
    </after> 
  </sample>
     ... all your samples ... 
</thinking>

Once you're prodced the candidate samples, go back through the list an select the ones that you think best represent the differces.  The transcript contains transcription errors, discard samples that appear to be transcription errors.  Note that the actual content in this instance is less important than the style, so look for rephrases that are distinctive.

Once you're done, format your answers like this:

<final>
  <sample>
    <boring>
      ... best version 1...
    </boring>
    <after>
      ... best sentence 1...
    </after> 
  </sample>
  <sample>
    <boring>
      ... best version 2...
    </boring>
    <after>
      ... best sentence 2...
    </after> 
  </sample>
     ... all your samples ... 
</final>