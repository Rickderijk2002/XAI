## Into / Tab 1 ##

S: Okay, so, the tool is going to show you a counterfactual, then ask about your own intuition, show a few metrics, then show some comparison so you can see the difference.

P: Okay

S: All feedback is welcome, it's (a) "thinking out loud" (test), (...)

P: Alright. Okay, welcome to the AI Counterfactual study... uhh it's kind of a lot of text

S: You don't have to read everything, most of it is a repeat of what I already explained (before the recording started).

P: ...Before you begin, read this page carefully, it explains what you will do, okay, so these are instructions.
P: Information; throughout the study you will see an info button on every page. Click on it to get a quick explanation relevant to that step.
P: You never need to come back to this page. Okay, that's good to know.
P: Hmmm... purpose of the study... research project at JADS... how human judgment of AI explanation compares to objective evaluation metrics.
P: Okay. Images of handwriten digits and modified versions called counterfactual explanations. These are images tweaked so that the AI model
P: predicts a different digit class than the original.
P: And so I have to... judge whether the counterfactuals... okay, clear. I think I get the goal.
P: Oh, here some explanation about counterfactuals, okay. Fine. A good counterfactual should fool the model into predicting the target class,
P: (...) still look realistic, okay. Is that always the case? That a counterfactual should be (realistic) for a human?

S: That is the ideal I would say yeah, if people are to use them.

P: Okay, yeah, if people are to use them indeed. Okay, so these then are further explanation; explanation about what each method does...
P: Okay that's... also interesting (...) okay, and then metrics, okay so this is the dataset description sort of.
P: Okay uhm... IM1 and implausibility go down when the CF is better, okay. 
P: What will happen during the study... okay, visual inspection, prediction, the game, metrics revealed, explanation and feedback, 
P: compare methods, final reflection. Okay, I think this page is clear now.

S: Okay, then you can enter a name and start.

P: Name or participant... okay...

S: You can enter anything, it doesn't really matter.

P: (enters name) okay. Press... start.

## Step 1

P: Okay, screen, this looks... visually... okay, it looks cohesive.
P: uhh what am I looking at (reading info box)... one original digit and one counterfactual. The counterfactual has been *modified*
P: so the AI model predicts a different class. Okay, "counterfactual has been modified". Does that mean that this is the original
P: counterfactual (image) or...?

S: Yeah it is just the original counterfactual image (from the dataset).

P: Original counterfactual, clear. (...) no metrics are shown yet on purpose. We want your raw visual intuition first. Okay.
P: Based on visual explanation, does this counterfactual look successful. Target 6. Uhh no, it does not look successful.
P: Uhh, next. Or yeah, I could real quick type here... "Counterfactual looks like an inverted 3 instead of 6".

## Step 2

P: Rate the following based on what you see. These are images... Okay validity do you think the model was fooled... Oh okay,
P: a slider... Oh that can... Do I have to... Okay completely... 0 definitely not, 1 definitely yes... I'm rather thinking
P: less than more, but how much I have to enter exactly... Well let's go for it, something like this? 30? Hmm, okay.
P: Plausibility, how realistic does this... No it doesn't look particularly realistic (fills in 0.06). Okay, next.

## Step 3

P: Okay, what do we have on the side here, oh this is an overview of the whole thing... guided task mode, mini-game,
P: oh and you can choose here, okay...
P: The game: choose the best explanation, okay information? Now you see all 5 methods side by side, consider two things:
P: does the counterfactual look like it belongs to the target class; does it look like a realistic image...
P: Which one do you think is most successful... consider both whether it looks realistic or whether the model would be fooled.
P: And this is still on...

S: It's still on the digit 9 to counterfactual 6.

P: 9 on 6, okay. (thinking). This is... Min-Edit... It's not very readable... Min-Edit. How confident are you in your choice...
P: Well I actually find all of them to be not very good. So we're gonna put this a bit lower... it's this one pixel here that's a
P: bit darker in the middle.
P: Oh yeah... "Darker pixel in the lower part of the image looks like the hole in the digit 6". ("next" button does not light up) Okay,
P: can I...

S: Yeah you just have to (click outside the text box)

P: Okay...

S: Yeah that's not ideal, I'll note that down.

## Step 4

P: Yep, okay, actual metrics revealed... before we show you the metrics here is a reminder of what each metric does, okay.
P: That's quite a lot of information. Metrics for the counterfactual shown in step (1 and 2), okay...
P: Human estimate, actual value... the model was NOT fooled by this counterfactual. Okay but this... validity is binary.
P: Why then did I get a slider? Okay, plausibility, your human estimate was (...), 1 is very realistic, around the dataset mean.
P: Plausibility, is that the second thing I just...?

S: That was the second metric you filled in, about how realistic it looks.

P: How realistic it looks, okay but that's..., plausibility human estimate versus... okay... I don't really know where...
P: this feels weird.
P: Implausibility score, uhm... 1 is very realistic, okay so according to... and this is the MNIST... where...
P: Are these the things I entered earlier?

S: The first two are the things you entered before, with the human estimate.

P: So this is for the MiniMax (Min-Edit) I just... or rather for the counterfactual I just found, or is it for this specific...

S: It's for this specific counterfactual.

P: This specific one, okay. Your reaction... yeah I'm a bit surprised. I'm not really sure where those metrics are coming from.
P: Or in any case, at first glance it doesn't really look very clear or intuitive.
P: "A bit confused not sure how the shown..." but alright this is the whole conflict between objective and subjective... "...are
P: derived and how this measures up to my subjective rating. Did I make a mistake here?"

## Step 5

P: CF produced by PIECE, okay. Alibi-proto, thats... remember this image, this is how you judged... you selected Min-Edit, okay.
P: Min-Edit was second... and on what is this (based)... IM1 lower is better right?

S: Yes.

P: Yeah (...) "what does ranking mean" right because this is based on IM1. Okay. And this is the best one or?

S: This is the one generated by PIECE.

P: Okay. That was the first one I was shown?
P: The objectively best method was alibi-proto-cf, but you chose Min-Edit. This difference is exactly what this study explores.
P: It would be nice to see... to be able to compare my choices with the best one they have here. Now I have to rely on my memory
P: and I don't fully remember how they differed.
P: Were you surprised... "I find it hard comment as I do not recall the two images exactly."

## Step 6

P: Okay next, compare all methods. So you have here that (comparison). And the Min-Edit was the best one...?

S: That was your pick.

P: Are there two Min-Edit's. Oh, this is C-Min-Edit. And PIECE was the best one?

S: PIECE was the original one (you got).

P: PIECE was the original, Min-Edit my choice, and...

S: I think alibi was the best one.

P: Alibi, okay... Oh yeah here it shows. What do these metrics mean, full evaluation of metrics... yeah PIECE,
P: Alibi-proto... okay that's that one. Okay, per-metric winner... most valid methods Min-Edit, best IM1 alibi-proto, okay.
P: Overall best method was alibi-proto (...) key insight best overall does not always mean best at everything. Okay.
P: One question this is required. Knowing the full metric breakdown, would you change your step... No I would not do that,
P: because alibi-proto to me still looks like a 9, with a 7 in the middle. Human eye, no.
P: "To me, alibi-proto does not look like the digit 6, so I would not pick it." Okay.

# Step 7

P: Okay these are my answers, summary, okay.
P: Uhm... overview, what should I reflect on here... think back to your first impressions (...) step 1.
P: Now you have seen the full rankings and metrics, does your initial judgment hold up, were you surprised by any of the results...
P: Well yeah, I'm somewhat surprised by the objective ratings. But it hasn't much... how confident are you in your step 1 judgment
P: now that you have seen everything. I'll put it at 50%, because the objective ratings have made it a bit more vague.
P: Would you change your step 1 answer after seeing all the metrics, yes I would change it but that's mainly because I would
P: understand better what I'm truly judging. And especially with the binary for example.
P: Okay (tries to end)

S: There's still a box here (pointing out "final thought" box)

P: Oh, final thoughts... did the metrics allign with your intuition overall. "No, this difference is what surprised me most."
P: Okay, results. Wow balloons wow.

S: Okay, do you have any further comments or anything?

P: Hmm, no, or well with regards to use, having to control-enter click after filling in text boxes is some extra effort.
P: It's doable, but it could be easier.

S: Hmm, okay. Well, great, thank you.